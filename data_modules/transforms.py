import math
import numbers
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from helpers.matching_utils import (convert_flow_to_mapping,
                                    convert_mapping_to_flow,
                                    create_border_mask,
                                    unnormalise_and_convert_mapping_to_flow,
                                    warp)
from PIL import Image, ImageEnhance, ImageFilter
from torch import Tensor
from torchvision.transforms import Compose

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)

pillow_interp_codes = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'box': Image.BOX,
    'lanczos': Image.LANCZOS,
    'hamming': Image.HAMMING
}


def get_image_size(sample: Dict, apply_keys: List[str]) -> List[int]:
    # assuming all images have same size
    for image_key in ['image', 'image_ref', 'image_prime']:
        if image_key in sample.keys() and (len(apply_keys) == 0 or image_key in apply_keys):
            break
    else:
        raise ValueError
    img = sample[image_key]
    if isinstance(img, Tensor):
        h, w = img.shape[-2:]
    else:
        w, h = img.size
    return w, h


def get_sample_device(sample: Dict):
    for image_key in ['image', 'image_ref', 'image_prime']:
        if image_key in sample.keys():
            break
    else:
        raise ValueError
    return sample[image_key].device


def imresize(img, size, interpolation='bilinear'):
    assert isinstance(img, Image.Image)
    if isinstance(size, Sequence) and len(size) == 1:
        size = size[0]
    if isinstance(size, int):
        w, h = img.size
        short, long = (w, h) if w <= h else (h, w)
        if short == size:
            return img

        new_short, new_long = size, int(size * long / short)
        new_w, new_h = (new_short, new_long) if w <= h else (
            new_long, new_short)
        return img.resize((new_w, new_h), pillow_interp_codes[interpolation])

    else:
        new_h, new_w = size
        return img.resize((new_w, new_h), pillow_interp_codes[interpolation])


def elastic_transform(shape, sigma, alpha, get_flow=False, approximate=True, device='cpu'):
    """ Apply an elastic distortion to the image
    https://github.com/albu/albumentations
    Parameters:
    sigma_params: sigma can vary between max(img.shape) * sigma_params[0] and
                    max(img.shape) * (sigma_params[0] + sigma_params[1])
    alpha_params: alpha can vary between max(img.shape) * alpha_params[0] and
                    max(img.shape) * (alpha_params[0] + alpha_params[1])
    """
    shape = shape[:2]
    [height, width] = shape

    # Create the grid
    if approximate:
        # adapted from OpenCV: https://github.com/opencv/opencv/blob/5cc154147f749c0d9ac7a32e4b12aa7469b817c3/modules/imgproc/src/smooth.dispatch.cpp#L289
        # k = int(max(round(sigma * 8) + 1, 1))
        # kx = min(k, width)
        # ky = min(k, height)
        # # make it odd
        # kx = math.ceil(kx / 2) * 2 - 1
        # ky = math.ceil(ky / 2) * 2 - 1
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        dx = torch.rand(height, width, dtype=torch.float,
                        device=device).numpy() * 2 - 1
        cv2.GaussianBlur(dx, (0, 0), sigma, dst=dx)
        # dx = torchvision.transforms.functional.gaussian_blur(
        #     dx.unsqueeze(0), kernel_size=[kx, ky], sigma=sigma).squeeze(0)
        dx = torch.from_numpy(dx)
        dx *= alpha

        dy = torch.rand(height, width, dtype=torch.float,
                        device=device).numpy() * 2 - 1
        cv2.GaussianBlur(dy, (0, 0), sigma, dst=dy)
        # dy = torchvision.transforms.functional.gaussian_blur(
        #     dy.unsqueeze(0), kernel_size=[kx, ky], sigma=sigma).squeeze(0)
        dy = torch.from_numpy(dy)
        dy *= alpha
    else:
        raise NotImplementedError

    if get_flow:
        return dx, dy

    else:
        x, y = torch.meshgrid(torch.arange(
            shape[1], dtype=torch.float, device=device), torch.arange(shape[0], dtype=torch.float, device=device), indexing='xy')
        # Apply the distortion
        map_x = x + dx
        map_y = y + dy
        return map_x, map_y


class Resize:

    def __init__(self, apply_keys='all', size=None, img_interpolation='bilinear', img_only=False, only_if_larger=False):
        assert isinstance(size, int) or (
            isinstance(size, Sequence) and len(size) == 2)
        self.apply_keys = apply_keys
        self.size = size
        self.img_interpolation = img_interpolation
        self.img_only = img_only
        self.only_if_larger = only_if_larger

    def __call__(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys

        if 'corr_pts' in apply_keys:
            w_pts, h_pts = get_image_size(sample, ['image'])
        if 'corr_pts_ref' in apply_keys:
            w_pts_ref, h_pts_ref = get_image_size(sample, ['image_ref'])

        if self.only_if_larger:
            w, h = get_image_size(sample, apply_keys)
            h_ratio = self.size[0] / h
            w_ratio = self.size[1] / w
            ratio = min(h_ratio, w_ratio)
            if ratio >= 1:
                return sample
            size = (int(round(ratio * h)), int(round(ratio * w)))

        else:
            size = self.size

        for key in apply_keys:
            val = sample[key]
            if key in ['image', 'image_ref', 'image_prime']:
                sample[key] = imresize(
                    val, size=size, interpolation=self.img_interpolation)
            elif key == 'semantic':
                if not self.img_only:
                    sample[key] = imresize(
                        val, size=size, interpolation='nearest')
            elif key == 'corr_pts':
                if not self.img_only:
                    pts = val
                    if isinstance(size, int):
                        short, long = (w_pts, h_pts) if w_pts <= h_pts else (
                            h_pts, w_pts)
                        if short == size:
                            continue
                        new_short, new_long = size, int(size * long / short)
                        new_w, new_h = (new_short, new_long) if w_pts <= h_pts else (
                            new_long, new_short)
                    else:
                        new_h, new_w = size
                    x_scale = new_w / float(w_pts)
                    y_scale = new_h / float(h_pts)
                    pts[:, 0] = x_scale * pts[:, 0]
                    pts[:, 1] = y_scale * pts[:, 1]
                    sample[key] = pts
            elif key == 'corr_pts_ref':
                if not self.img_only:
                    pts = val
                    if isinstance(size, int):
                        short, long = (w_pts_ref, h_pts_ref) if w_pts_ref <= h_pts_ref else (
                            h_pts_ref, w_pts_ref)
                        if short == size:
                            continue
                        new_short, new_long = size, int(size * long / short)
                        new_w, new_h = (new_short, new_long) if w_pts_ref <= h_pts_ref else (
                            new_long, new_short)
                    else:
                        new_h, new_w = size
                    x_scale = new_w / float(w_pts_ref)
                    y_scale = new_h / float(h_pts_ref)
                    pts[:, 0] = x_scale * pts[:, 0]
                    pts[:, 1] = y_scale * pts[:, 1]
                    sample[key] = pts
            elif key in ['filename', 'image_prime_idx']:
                pass
            else:
                raise ValueError
        return sample


class RandomRotation(torchvision.transforms.RandomRotation):

    def __init__(self, apply_keys='all', **kwargs):
        super().__init__(**kwargs)
        self.apply_keys = apply_keys

    def rotate(self, img, angle, fill=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.
        Returns:
            PIL Image or Tensor: Rotated image.
        """
        assert isinstance(img, Image.Image)
        if fill is None:
            fill = self.fill

        return torchvision.transforms.functional.rotate(img, angle, self.resample, self.expand, self.center, fill)

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys

        angle = self.get_params(self.degrees)

        for key in apply_keys:
            val = sample[key]
            if key in ['image', 'image_ref', 'image_prime']:
                sample[key] = self.rotate(val, angle=angle)
            elif key in ['semantic', 'semantic']:
                sample[key] = self.rotate(val, angle=angle, fill=255)
            elif key in ['filename', 'image_prime_idx']:
                pass
            else:
                raise ValueError

        w, h = get_image_size(sample, apply_keys)
        sample['normalize_mask'] = self.rotate(
            Image.new("1", (w, h), 0), angle=angle, fill=1)
        return sample


class ToTensor:

    def __init__(self, apply_keys='all'):
        self.apply_keys = apply_keys

    def __call__(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        elif self.apply_keys == 'none':
            apply_keys = []
        else:
            apply_keys = self.apply_keys

        for key in apply_keys:
            val = sample[key]
            if key in ['image', 'image_ref', 'image_prime', 'normalize_mask']:
                sample[key] = torchvision.transforms.functional.pil_to_tensor(
                    val)
            elif key == 'semantic':
                sample[key] = torchvision.transforms.functional.pil_to_tensor(
                    val).squeeze(0)
            elif key in ['filename', 'image_prime_idx', 'corr_pts', 'corr_pts_ref']:
                pass
            else:
                print(key)
                raise ValueError

        return sample


######## TORCH MODULES ############

class RandomCrop(nn.Module):
    def __init__(self, apply_keys='all', size=None, ignore_index=255, cat_max_ratio=1.0):
        super().__init__()
        self.apply_keys = apply_keys
        self.size = size
        self.ignore_index = ignore_index
        self.cat_max_ratio = cat_max_ratio

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys

        w, h = get_image_size(sample, apply_keys)
        crop_params = self.get_params([h, w], self.size)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_tmp = self.crop(sample['semantic'], *crop_params)
                labels, cnt = torch.unique(seg_tmp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and cnt.max() / torch.sum(cnt).float() < self.cat_max_ratio:
                    break
                crop_params = self.get_params([h, w], self.size)
        for key in apply_keys:
            val = sample[key]
            if key in ['image', 'image_ref', 'image_prime', 'semantic', 'normalize_mask']:
                sample[key] = self.crop(val, *crop_params)
            elif key in ['corr_pts']:
                pts1 = sample['corr_pts_ref']
                pts2 = sample['corr_pts']
                pts1, pts2 = self.adjust_correspondences(
                    pts1, pts2, crop_params)
                sample['corr_pts_ref'] = pts1
                sample['corr_pts'] = pts2
            elif key in ['filename', 'image_prime_idx', 'corr_pts_ref']:
                pass
            else:
                raise ValueError
        return sample

    def adjust_correspondences(self, pts1, pts2, crop_params):
        top, left, height, width = crop_params
        pts1[:, 0] = pts1[:, 0] - left
        pts1[:, 1] = pts1[:, 1] - top
        pts2[:, 0] = pts2[:, 0] - left
        pts2[:, 1] = pts2[:, 1] - top

        # remove all correspondence not seen in cropped image
        in_im = ((torch.round(pts1[:, 0]) >= 0) & (torch.round(pts1[:, 0]) < width) &
                 (torch.round(pts2[:, 0]) >= 0) & (torch.round(pts2[:, 0]) < width) &
                 (torch.round(pts1[:, 1]) >= 0) & (torch.round(pts1[:, 1]) < height) &
                 (torch.round(pts2[:, 1]) >= 0) & (torch.round(pts2[:, 1]) < height))
        pts1 = pts1[in_im]
        pts2 = pts2[in_im]
        return pts1, pts2

    @staticmethod
    def get_params(img_size, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img_size
        th, tw = output_size

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, max(h - th, 0))
        j = random.randint(0, max(w - tw, 0))
        return i, j, min(th, h), min(tw, w)

    def crop(self, img, top, left, height, width):
        h, w = img.shape[-2:]
        right = left + width
        bottom = top + height

        if left < 0 or top < 0 or right > w or bottom > h:
            raise ValueError("Invalid crop parameters: {}, img size: {}".format(
                (top, left, height, width), (h, w)))
        return img[..., top:bottom, left:right]


class RandomHorizontalFlip(nn.Module):
    def __init__(self, apply_keys='all', p=0.5):
        super().__init__()
        self.apply_keys = apply_keys
        self.p = p

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys

        if random.random() < self.p:
            for key in apply_keys:
                val = sample[key]
                if key in ['image', 'image_ref', 'image_prime', 'semantic', 'normalize_mask']:
                    sample[key] = torchvision.transforms.functional.hflip(val)
                elif key in ['corr_pts', 'corr_pts_ref']:
                    w, h = get_image_size(sample, apply_keys)
                    pts = sample[key]
                    pts[:, 0] = w - 1 - pts[:, 0]
                    sample[key] = pts
                elif key in ['filename', 'image_prime_idx']:
                    pass
                else:
                    raise ValueError

        return sample


class ColorJitter(torchvision.transforms.ColorJitter):

    def __init__(self, apply_keys='all', **kwargs):
        super().__init__(**kwargs)
        self.apply_keys = apply_keys

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys

        for key in apply_keys:
            val = sample[key]
            if key in ['image', 'image_ref', 'image_prime']:
                sample[key] = super().forward(val)
            elif key in ['semantic', 'corr_pts', 'corr_pts_ref', 'filename', 'image_prime_idx', 'normalize_mask']:
                pass
            else:
                raise ValueError
        return sample


class ChannelShuffle(nn.Module):

    def __init__(self, apply_keys='all'):
        super().__init__()
        self.apply_keys = apply_keys

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys
        for key in apply_keys:
            val = sample[key]
            if key == 'image_prime':
                indices = list(range(3))
                random.shuffle(indices)
                sample[key] = val[indices]
            else:
                raise ValueError
        return sample


class ConvertImageDtype(torchvision.transforms.ConvertImageDtype):

    def __init__(self, apply_keys='all', scaling=True, **kwargs):
        dtype = kwargs.pop('dtype', torch.float)
        super().__init__(dtype=dtype, **kwargs)
        self.apply_keys = apply_keys
        self.scaling = scaling

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys
        for key in apply_keys:
            val = sample[key]
            if key in ['image', 'image_ref', 'image_prime']:
                if not self.scaling:
                    sample[key] = val.to(self.dtype)
                else:
                    sample[key] = super().forward(val)
            elif key == 'semantic':
                sample[key] = val.to(torch.long)  # from byte to long
            elif key in ['corr_pts', 'corr_pts_ref', 'filename', 'image_prime_idx', 'normalize_mask']:
                pass
            else:
                raise ValueError
        return sample


class Normalize(torchvision.transforms.Normalize):
    def __init__(self, apply_keys='all', **kwargs):
        # set imagenet statistics as default
        mean = kwargs.pop('mean', IMNET_MEAN)
        std = kwargs.pop('std', IMNET_STD)
        super().__init__(mean=mean, std=std, **kwargs)
        self.apply_keys = apply_keys

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys

        for key in apply_keys:
            val = sample[key]
            if key in ['image', 'image_ref', 'image_prime']:
                normalized = super().forward(val)
                if 'normalize_mask' in sample.keys():
                    normalized[sample['normalize_mask'].expand_as(
                        normalized)] = 0.0
                sample[key] = normalized
                sample[key] = normalized
            elif key in ['semantic', 'corr_pts', 'corr_pts_ref', 'filename', 'image_prime_idx', 'normalize_mask']:
                pass
            else:
                raise ValueError
        sample.pop('normalize_mask', None)
        return sample


class RandomGaussianBlur(torchvision.transforms.GaussianBlur):

    def __init__(self, apply_keys='all', p=0.2, **kwargs):
        super().__init__(**kwargs)
        self.apply_keys = apply_keys
        self.p = p

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys
        for key in apply_keys:
            val = sample[key]
            if key in ['image', 'image_ref', 'image_prime']:
                if random.random() < self.p:
                    sample[key] = super().forward(val)
            elif key in ['semantic', 'corr_pts', 'corr_pts_ref', 'filename', 'image_prime_idx']:
                pass
            else:
                raise ValueError
        return sample


class PadBottomRight(nn.Module):
    def __init__(self, apply_keys='all', size=None, same_shape_keys=None, ignore_index=255):
        super().__init__()
        self.apply_keys = apply_keys
        self.size = size
        self.same_shape_keys = same_shape_keys
        if self.size is None:
            assert self.same_shape_keys is not None
            assert len(self.same_shape_keys) == 2
        if self.same_shape_keys is None:
            assert self.size is not None
        self.ignore_index = ignore_index

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys
        if self.same_shape_keys is not None:
            w1, h1 = get_image_size(sample, [self.same_shape_keys[0]])
            w2, h2 = get_image_size(sample, [self.same_shape_keys[1]])
            h_final = max(h1, h2)
            w_final = max(w1, w2)
        elif self.aspect_ratio is not None:
            w, h = get_image_size(sample, apply_keys)
            w_final = int(round(max(w, h / float(self.aspect_ratio))))
            h_final = int(round(max(h, w * float(self.aspect_ratio))))
        else:
            h_final, w_final = self.size
        for key in apply_keys:
            val = sample[key]
            if key in ['image', 'image_ref', 'image_prime']:
                sample[key] = self.pad(val, h_final, w_final, fill=0)
            elif key == 'semantic':
                sample[key] = self.pad(
                    val, h_final, w_final, fill=self.ignore_index)
            elif key in ['filename', 'image_prime_idx', 'corr_pts', 'corr_pts_ref']:
                pass
            else:
                raise ValueError
        return sample

    def pad(self, img, new_h, new_w, fill=0):
        h, w = img.shape[-2:]
        if h == new_h and w == new_w:
            return img
        bottom_pad = new_h - h
        right_pad = new_w - w
        return nn.functional.pad(img, (0, right_pad, 0, bottom_pad), value=fill)


class RandomAffine(nn.Module):

    def __init__(self,
                 apply_keys='all',
                 random_alpha=0.065,
                 random_s=0.6,
                 random_tx=0.3,
                 random_ty=0.1,
                 parameterize_with_gaussian=False,
                 preserve_aspect_ratio=True,
                 min_fraction_valid_corr=0.1,
                 return_flow=False):
        super().__init__()
        self.apply_keys = apply_keys
        self.random_alpha = random_alpha
        self.random_s = random_s
        self.random_tx = random_tx
        self.random_ty = random_ty
        self.parameterize_with_gaussian = parameterize_with_gaussian
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.min_fraction_valid_corr = min_fraction_valid_corr
        self.return_flow = return_flow

    @staticmethod
    def get_params(random_alpha, random_s, random_tx, random_ty, out_h, out_w, parameterize_with_gaussian, preserve_aspect_ratio, device):
        if parameterize_with_gaussian:
            rot_angle = torch.empty(1).normal_(0, random_alpha).item()
            sh_angle = torch.empty(1).normal_(0, random_alpha).item()
            lambda_1 = torch.empty(1).normal_(1, random_s).item()
            if preserve_aspect_ratio:
                lambda_2 = lambda_1
            else:
                lambda_2 = torch.empty(1).normal_(1, random_s).item()
            tx = torch.empty(1, device=device).normal_(0, random_tx)
            ty = torch.empty(1, device=device).normal_(0, random_ty)
        else:
            rot_angle = (torch.rand(1).item() - 0.5) * 2 * random_alpha
            # between -np.pi/12 and np.pi/12 for random_alpha = np.pi/12
            sh_angle = (torch.rand(1).item() - 0.5) * 2 * random_alpha
            # between 0.75 and 1.25 for random_s = 0.25
            lambda_1 = 1 + (2 * torch.rand(1).item() - 1) * random_s
            if preserve_aspect_ratio:
                lambda_2 = lambda_1
            else:
                lambda_2 = 1 + (2 * torch.rand(1).item() - 1) * \
                    random_s  # between 0.75 and 1.25
            # between -0.25 and 0.25 for random_t=0.25
            tx = (2 * torch.rand(1, device=device) - 1) * random_tx
            ty = (2 * torch.rand(1, device=device) - 1) * random_ty

        R_sh = torch.tensor([[math.cos(sh_angle), -math.sin(sh_angle)],
                            [math.sin(sh_angle), math.cos(sh_angle)]], device=device)
        R_alpha = torch.tensor([[math.cos(rot_angle), -math.sin(rot_angle)],
                                [math.sin(rot_angle), math.cos(rot_angle)]], device=device)

        D = torch.diag(torch.tensor([lambda_1, lambda_2], device=device))
        A = R_alpha @ R_sh.T @ D @ R_sh

        theta_aff = torch.stack(
            [A[0, 0], A[0, 1], tx[0], A[1, 0], A[1, 1], ty[0]]).unsqueeze(0)

        if not theta_aff.size() == (1, 2, 3):
            theta_aff = theta_aff.view(1, 2, 3)

        return nn.functional.affine_grid(theta_aff, [1, 3, out_h, out_w], align_corners=False)

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys

        w, h = get_image_size(sample, apply_keys)
        device = get_sample_device(sample)

        mapping = self.get_params(self.random_alpha,
                                  self.random_s,
                                  self.random_tx,
                                  self.random_ty,
                                  h,
                                  w,
                                  self.parameterize_with_gaussian,
                                  self.preserve_aspect_ratio,
                                  device)
        flow_gt = unnormalise_and_convert_mapping_to_flow(
            mapping, output_channel_first=True)
        if self.return_flow:
            return flow_gt

        for key in apply_keys:
            val = sample[key]
            if key == 'image_prime':
                sample[key], sample[key + '_flow'], sample[key +
                                                           '_mask'] = self.apply_transform(val, flow_gt)
            else:
                raise ValueError
        return sample

    def apply_transform(self, image, flow_gt):
        # get synthetic homography transformation from the synthetic flow generator
        # flow_gt here is between target prime and target
        assert image.shape[-2:] == flow_gt.shape[-2:]
        if image.ndim == 3:
            image.unsqueeze_(0)
        image_prime, mask = warp(
            image, flow_gt, padding_mode='zeros', return_mask=True)
        # ground truth correspondence mask for flow between target prime and target
        mask_corr_gt = create_border_mask(flow_gt)
        # if mask_gt has too little valid areas, overwrite to use that mask in anycase
        if mask_corr_gt.sum() < mask_corr_gt.shape[-1] * mask_corr_gt.shape[-2] * self.min_fraction_valid_corr:
            mask = mask_corr_gt
        return image_prime.squeeze(0), flow_gt.squeeze(0), mask.squeeze(0)


class RandomHomography(nn.Module):

    def __init__(self,
                 apply_keys='all',
                 random_t_hom=0.3,
                 parameterize_with_gaussian=False,
                 add_elastic=False,
                 min_fraction_valid_corr=0.1,
                 return_flow=False):
        super().__init__()
        self.apply_keys = apply_keys
        self.random_t_hom = random_t_hom
        self.parameterize_with_gaussian = parameterize_with_gaussian
        self.min_fraction_valid_corr = min_fraction_valid_corr
        self.return_flow = return_flow

    @staticmethod
    def get_params(random_t_hom, out_h, out_w, parameterize_with_gaussian, device):
        theta = torch.tensor([-1, -1, 1, 1, -1, 1, -1, 1],
                             device=device, dtype=torch.float)
        if parameterize_with_gaussian:
            theta = theta + \
                torch.empty(8, device=device).normal_(0, random_t_hom)
        else:
            theta = theta + (torch.rand(8, device=device) -
                             0.5) * 2 * random_t_hom
        theta = theta.unsqueeze(0)

        # create homography matrix from 4 pts
        b = theta.size(0)
        xp = theta[:, :4].unsqueeze(2)
        yp = theta[:, 4:].unsqueeze(2)

        x = theta.new_tensor(
            [-1, -1, 1, 1]).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
        y = theta.new_tensor(
            [-1, 1, -1, 1]).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
        z = theta.new_zeros(4).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
        o = theta.new_ones(4).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
        single_o = theta.new_ones(1).unsqueeze(1).unsqueeze(0).expand(b, 1, 1)

        A = torch.cat([torch.cat([-x, -y, -o, z, z, z, x * xp, y * xp, xp], 2),
                       torch.cat([z, z, z, -x, -y, -o, x * yp, y * yp, yp], 2)], 1)
        # find homography by assuming h33 = 1 and inverting the linear system
        h = torch.bmm(torch.inverse(A[:, :, :8]), -A[:, :, 8].unsqueeze(2))
        # add h33
        h = torch.cat([h, single_o], 1)
        H = h.squeeze(2)

        # get grid
        h0 = H[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h1 = H[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h2 = H[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h3 = H[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h4 = H[:, 4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h5 = H[:, 5].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h6 = H[:, 6].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h7 = H[:, 7].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h8 = H[:, 8].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        def expand_dim(tensor, dim, desired_dim_len):
            sz = list(tensor.size())
            sz[dim] = desired_dim_len
            return tensor.expand(tuple(sz))

        grid_X, grid_Y = torch.meshgrid(
            torch.linspace(-1, 1, out_w, device=device), torch.linspace(-1, 1, out_h, device=device))
        grid_X = grid_X.T.unsqueeze(0).unsqueeze(3)
        grid_Y = grid_Y.T.unsqueeze(0).unsqueeze(3)
        grid_X = expand_dim(grid_X, 0, b)
        grid_Y = expand_dim(grid_Y, 0, b)

        grid_Xp = grid_X * h0 + grid_Y * h1 + h2
        grid_Yp = grid_X * h3 + grid_Y * h4 + h5
        k = grid_X * h6 + grid_Y * h7 + h8

        grid_Xp /= k
        grid_Yp /= k

        return torch.cat((grid_Xp, grid_Yp), 3)

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys
        w, h = get_image_size(sample, apply_keys)
        device = get_sample_device(sample)

        mapping = self.get_params(
            self.random_t_hom, h, w, self.parameterize_with_gaussian, device)
        flow_gt = unnormalise_and_convert_mapping_to_flow(
            mapping, output_channel_first=True)
        if self.return_flow:
            return flow_gt

        for key in apply_keys:
            val = sample[key]
            if key == 'image_prime':
                sample[key], sample[key + '_flow'], sample[key +
                                                           '_mask'] = self.apply_transform(val, flow_gt)
            else:
                raise ValueError
        return sample

    def apply_transform(self, image, flow_gt):
        # get synthetic homography transformation from the synthetic flow generator
        # flow_gt here is between target prime and target
        assert image.shape[-2:] == flow_gt.shape[-2:]
        if image.ndim == 3:
            image.unsqueeze_(0)
        image_prime, mask = warp(
            image, flow_gt, padding_mode='zeros', return_mask=True)
        # ground truth correspondence mask for flow between target prime and target
        mask_corr_gt = create_border_mask(flow_gt)
        # if mask_gt has too little valid areas, overwrite to use that mask in anycase
        if mask_corr_gt.sum() < mask_corr_gt.shape[-1] * mask_corr_gt.shape[-2] * self.min_fraction_valid_corr:
            mask = mask_corr_gt
        return image_prime.squeeze(0), flow_gt.squeeze(0), mask.squeeze(0)


class RandomTPS(nn.Module):

    def __init__(self,
                 apply_keys='all',
                 random_t_tps=0.3,
                 grid_size=3,
                 reg_factor=0,
                 parameterize_with_gaussian=False,
                 add_elastic=False,
                 min_fraction_valid_corr=0.1,
                 return_flow=False):
        super().__init__()
        self.apply_keys = apply_keys
        self.random_t_tps = random_t_tps
        self.parameterize_with_gaussian = parameterize_with_gaussian
        self.min_fraction_valid_corr = min_fraction_valid_corr
        self.return_flow = return_flow

        axis_coords = np.linspace(-1, 1, grid_size)
        self.N = grid_size * grid_size
        P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
        P_X = np.reshape(P_X, (-1, 1))  # load_size (N,1)
        P_Y = np.reshape(P_Y, (-1, 1))  # load_size (N,1)
        P_X = torch.FloatTensor(P_X)
        P_Y = torch.FloatTensor(P_Y)
        self.register_buffer('Li', self.compute_L_inverse(
            P_X, P_Y, reg_factor).unsqueeze(0))
        self.register_buffer('P_X', P_X.unsqueeze(
            2).unsqueeze(3).unsqueeze(4).transpose(0, 4))
        self.register_buffer('P_Y', P_Y.unsqueeze(
            2).unsqueeze(3).unsqueeze(4).transpose(0, 4))

    @staticmethod
    def get_params(random_t_tps, out_h, out_w, N, Li, P_X, P_Y, parameterize_with_gaussian, device):
        theta = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, 0,
                             1, -1, 0, 1, -1, 0, 1], device=device, dtype=torch.float)
        if parameterize_with_gaussian:
            theta = theta + \
                torch.empty(18, device=device).normal_(0, random_t_tps)
        else:
            theta = theta + (torch.rand(18, device=device) -
                             0.5) * 2 * random_t_tps
        theta = theta.unsqueeze(0)

        grid_X, grid_Y = torch.meshgrid(
            torch.linspace(-1, 1, out_w, device=device), torch.linspace(-1, 1, out_h, device=device))
        grid_X = grid_X.T.unsqueeze(0).unsqueeze(3)
        grid_Y = grid_Y.T.unsqueeze(0).unsqueeze(3)
        points = torch.cat((grid_X, grid_Y), 3)

        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :N, :, :].squeeze(3)
        Q_Y = theta[:, N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = P_X.expand((1, points_h, points_w, 1, N))
        P_Y = P_Y.expand((1, points_h, points_w, 1, N))

        # compute weigths for non-linear part
        W_X = torch.bmm(Li[:, :N, :N].expand((batch_size, N, N)), Q_X)
        W_Y = torch.bmm(Li[:, :N, :N].expand((batch_size, N, N)), Q_Y)
        # reshape
        # W_X,W,Y: load_size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(
            1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(
            1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        A_X = torch.bmm(Li[:, N:, :N].expand((batch_size, 3, N)), Q_X)
        A_Y = torch.bmm(Li[:, N:, :N].expand((batch_size, 3, N)), Q_Y)
        # reshape
        # A_X,A,Y: load_size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(
            1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(
            1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 0].size() + (1, N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 1].size() + (1, N))

        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - \
                P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - \
                P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: load_size [1,H,W,1,N]
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand(
                (batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand(
                (batch_size,) + points_Y_batch.size()[1:])

        points_X_prime = A_X[:, :, :, :, 0] + \
            torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
            torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        warped_grid = torch.cat((points_X_prime, points_Y_prime), 3).float()
        return warped_grid

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys
        w, h = get_image_size(sample, apply_keys)
        device = get_sample_device(sample)

        mapping = self.get_params(self.random_t_tps,
                                  h,
                                  w,
                                  self.N,
                                  self.Li,
                                  self.P_X,
                                  self.P_Y,
                                  self.parameterize_with_gaussian,
                                  device)
        flow_gt = unnormalise_and_convert_mapping_to_flow(
            mapping, output_channel_first=True)
        if self.return_flow:
            return flow_gt

        for key in apply_keys:
            val = sample[key]
            if key == 'image_prime':
                sample[key], sample[key + '_flow'], sample[key +
                                                           '_mask'] = self.apply_transform(val, flow_gt)
            else:
                raise ValueError
        return sample

    def apply_transform(self, image, flow_gt):
        # get synthetic homography transformation from the synthetic flow generator
        # flow_gt here is between target prime and target
        assert image.shape[-2:] == flow_gt.shape[-2:]
        if image.ndim == 3:
            image.unsqueeze_(0)
        image_prime, mask = warp(
            image, flow_gt, padding_mode='zeros', return_mask=True)
        # ground truth correspondence mask for flow between target prime and target
        mask_corr_gt = create_border_mask(flow_gt)
        # if mask_gt has too little valid areas, overwrite to use that mask in anycase
        if mask_corr_gt.sum() < mask_corr_gt.shape[-1] * mask_corr_gt.shape[-2] * self.min_fraction_valid_corr:
            mask = mask_corr_gt
        return image_prime.squeeze(0), flow_gt.squeeze(0), mask.squeeze(0)

    @staticmethod
    def compute_L_inverse(X, Y, reg_factor):
        N = X.size()[0]  # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(
            Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        # make diagonal 1 to avoid NaN in log computation
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        if reg_factor != 0:
            K += torch.eye(K.size(0), K.size(1)) * reg_factor
        # construct matrix L
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat(
            (P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        return Li


class RandomAffineTPS(nn.Module):

    def __init__(self,
                 apply_keys='all',
                 random_alpha=0.065,
                 random_s=0.6,
                 random_tx=0.3,
                 random_ty=0.1,
                 random_t_tps_for_afftps=0,
                 grid_size=3,
                 reg_factor=0,
                 preserve_aspect_ratio=True,
                 parameterize_with_gaussian=False,
                 min_fraction_valid_corr=0.1,
                 return_flow=False):
        super().__init__()
        self.apply_keys = apply_keys
        self.random_t_tps_for_afftps = random_t_tps_for_afftps
        self.random_alpha = random_alpha
        self.random_s = random_s
        self.random_tx = random_tx
        self.random_ty = random_ty
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.parameterize_with_gaussian = parameterize_with_gaussian
        self.min_fraction_valid_corr = min_fraction_valid_corr
        self.return_flow = return_flow

        axis_coords = np.linspace(-1, 1, grid_size)
        self.N = grid_size * grid_size
        P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
        P_X = np.reshape(P_X, (-1, 1))  # load_size (N,1)
        P_Y = np.reshape(P_Y, (-1, 1))  # load_size (N,1)
        P_X = torch.FloatTensor(P_X)
        P_Y = torch.FloatTensor(P_Y)
        self.register_buffer('Li', RandomTPS.compute_L_inverse(
            P_X, P_Y, reg_factor).unsqueeze(0))
        self.register_buffer('P_X', P_X.unsqueeze(
            2).unsqueeze(3).unsqueeze(4).transpose(0, 4))
        self.register_buffer('P_Y', P_Y.unsqueeze(
            2).unsqueeze(3).unsqueeze(4).transpose(0, 4))

    @staticmethod
    def get_params(sampling_grid_aff, sampling_grid_aff_tps):
        # put 1e10 value in region out of bounds of sampling_grid_aff
        in_bound_mask_aff = ((sampling_grid_aff[:, :, :, 0] > -1) * (sampling_grid_aff[:, :, :, 0] < 1) * (
            sampling_grid_aff[:, :, :, 1] > -1) * (sampling_grid_aff[:, :, :, 1] < 1)).unsqueeze(3)
        in_bound_mask_aff = in_bound_mask_aff.expand_as(sampling_grid_aff)
        sampling_grid_aff = torch.mul(
            in_bound_mask_aff.float(), sampling_grid_aff)
        sampling_grid_aff = torch.add(
            (in_bound_mask_aff.float() - 1) * (1e10), sampling_grid_aff)

        # compose transformations
        sampling_grid_aff_tps_comp = nn.functional.grid_sample(sampling_grid_aff.transpose(2, 3).transpose(1, 2),
                                                               sampling_grid_aff_tps, align_corners=True)\
            .transpose(1, 2).transpose(2, 3)

        # put 1e10 value in region out of bounds of sampling_grid_aff_tps_comp
        in_bound_mask_aff_tps = ((sampling_grid_aff_tps[:, :, :, 0] > -1) * (sampling_grid_aff_tps[:, :, :, 0] < 1) * (
            sampling_grid_aff_tps[:, :, :, 1] > -1) * (sampling_grid_aff_tps[:, :, :, 1] < 1)).unsqueeze(3)
        in_bound_mask_aff_tps = in_bound_mask_aff_tps.expand_as(
            sampling_grid_aff_tps_comp)
        sampling_grid_aff_tps_comp = torch.mul(
            in_bound_mask_aff_tps.float(), sampling_grid_aff_tps_comp)
        sampling_grid_aff_tps_comp = torch.add(
            (in_bound_mask_aff_tps.float() - 1) * (1e10), sampling_grid_aff_tps_comp)
        return sampling_grid_aff_tps_comp

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys
        w, h = get_image_size(sample, apply_keys)
        device = get_sample_device(sample)

        affine_grid = RandomAffine.get_params(self.random_alpha,
                                              self.random_s,
                                              self.random_tx,
                                              self.random_ty,
                                              h,
                                              w,
                                              self.parameterize_with_gaussian,
                                              self.preserve_aspect_ratio,
                                              device)
        tps_grid = RandomTPS.get_params(self.random_t_tps_for_afftps,
                                        h,
                                        w,
                                        self.N,
                                        self.Li,
                                        self.P_X,
                                        self.P_Y,
                                        self.parameterize_with_gaussian,
                                        device)
        mapping = self.get_params(affine_grid, tps_grid)
        flow_gt = unnormalise_and_convert_mapping_to_flow(
            mapping, output_channel_first=True)
        if self.return_flow:
            return flow_gt

        for key in apply_keys:
            val = sample[key]
            if key == 'image_prime':
                sample[key], sample[key + '_flow'], sample[key +
                                                           '_mask'] = self.apply_transform(val, flow_gt)
            else:
                raise ValueError
        return sample

    def apply_transform(self, image, flow_gt):
        # get synthetic homography transformation from the synthetic flow generator
        # flow_gt here is between target prime and target
        assert image.shape[-2:] == flow_gt.shape[-2:]
        if image.ndim == 3:
            image.unsqueeze_(0)
        image_prime, mask = warp(
            image, flow_gt, padding_mode='zeros', return_mask=True)
        # ground truth correspondence mask for flow between target prime and target
        mask_corr_gt = create_border_mask(flow_gt)
        # if mask_gt has too little valid areas, overwrite to use that mask in anycase
        if mask_corr_gt.sum() < mask_corr_gt.shape[-1] * mask_corr_gt.shape[-2] * self.min_fraction_valid_corr:
            mask = mask_corr_gt
        return image_prime.squeeze(0), flow_gt.squeeze(0), mask.squeeze(0)


class RandomElastic(nn.Module):

    def __init__(self,
                 apply_keys='all',
                 min_nbr_perturbations=5,
                 max_nbr_perturbations=13,
                 min_sigma_mask=10,
                 max_sigma_mask=40,
                 min_sigma=0.1,
                 max_sigma=0.08,
                 min_alpha=1,
                 max_alpha=1,
                 min_fraction_valid_corr=0.1,
                 return_flow=False):
        super().__init__()
        self.apply_keys = apply_keys
        self.min_nbr_perturbations = min_nbr_perturbations
        self.max_nbr_perturbations = max_nbr_perturbations
        self.min_sigma_mask = min_sigma_mask
        self.max_sigma_mask = max_sigma_mask
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_fraction_valid_corr = min_fraction_valid_corr
        self.return_flow = return_flow

    @staticmethod
    def get_params(flow,
                   min_nbr_perturbations,
                   max_nbr_perturbations,
                   min_sigma_mask,
                   max_sigma_mask,
                   min_sigma,
                   max_sigma,
                   min_alpha,
                   max_alpha,
                   out_h,
                   out_w,
                   device):
        if flow is None:
            # no previous flow, it will be only elastic transformations
            flow = torch.zeros(1, out_h, out_w, 2,
                               device=device, dtype=torch.float)

        b = flow.shape[0]
        mapping = convert_flow_to_mapping(
            flow, output_channel_first=False)  # b, h, w, 2
        shape = list(mapping.shape[1:-1])

        # do the same synthetic transfo for the whole batch
        nbr_perturbations = random.randint(
            min_nbr_perturbations, max_nbr_perturbations)

        # sample parameters of elastic transform
        sigma_ = max(shape) * (min_sigma + max_sigma * random.random())
        alpha = max(shape) * (min_alpha + max_alpha * random.random())

        # get the elastic transformation
        flow_x_pertu, flow_y_pertu = elastic_transform(
            shape, sigma_, alpha, get_flow=True, approximate=True, device=device)
        flow_pertu = torch.stack((flow_x_pertu, flow_y_pertu), dim=2)
        mask_final = torch.zeros(
            shape, dtype=torch.float, device=flow_pertu.device)

        def gaussian_fn(M, mu, std):
            n = torch.arange(0, M)
            sig2 = 2 * std * std
            w = torch.exp(-(n - mu) ** 2 / sig2)
            return w

        def gkern(shape, mu, std):
            """Returns a 2D Gaussian kernel array."""
            gkern1d1 = gaussian_fn(shape[0], mu=mu[0], std=std)
            gkern1d2 = gaussian_fn(shape[1], mu=mu[1], std=std)
            gkern2d = torch.outer(gkern1d1, gkern1d2)
            div = std * math.sqrt(2 * math.pi) ** 2
            return gkern2d / div

        # make the mask
        for i in range(nbr_perturbations):
            sigma = random.randint(min_sigma_mask, max_sigma_mask)
            x = random.randint(0 + sigma * 3, shape[1] - sigma * 3)
            y = random.randint(0 + sigma * 3, shape[0] - sigma * 3)

            mask = gkern(shape, mu=[x, y], std=sigma)

            m = mask.max()
            if m < 1e-6:
                continue
            mask = torch.clamp(2.0 / m * mask, 0.0, 1.0)
            mask_final = mask_final + mask

        mask = torch.clamp(mask_final, 0.0, 1.0)
        # estimation final perturbation, shape is h,w,2
        flow_pertu = flow_pertu * mask.unsqueeze(2)
        flow_perturbation = flow_pertu.unsqueeze(0).expand(b, -1, -1, -1)

        # get final composition
        final_mapping = warp(mapping.permute(0, 3, 1, 2),
                             flow_perturbation.permute(0, 3, 1, 2))
        return final_mapping

    def forward(self, sample, flow=None):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys
        w, h = get_image_size(sample, apply_keys)
        device = get_sample_device(sample)

        mapping = self.get_params(flow,
                                  self.min_nbr_perturbations,
                                  self.max_nbr_perturbations,
                                  self.min_sigma_mask,
                                  self.max_sigma_mask,
                                  self.min_sigma,
                                  self.max_sigma,
                                  self.min_alpha,
                                  self.max_alpha,
                                  h,
                                  w,
                                  device)
        flow_gt = convert_mapping_to_flow(mapping, output_channel_first=True)
        if self.return_flow:
            return flow_gt

        for key in apply_keys:
            val = sample[key]
            if key == 'image_prime':
                sample[key], sample[key + '_flow'], sample[key +
                                                           '_mask'] = self.apply_transform(val, flow_gt)
            else:
                raise ValueError
        return sample

    def apply_transform(self, image, flow_gt):
        # get synthetic homography transformation from the synthetic flow generator
        # flow_gt here is between target prime and target
        assert image.shape[-2:] == flow_gt.shape[-2:]
        if image.ndim == 3:
            image.unsqueeze_(0)
        image_prime, mask = warp(
            image, flow_gt, padding_mode='zeros', return_mask=True)
        # ground truth correspondence mask for flow between target prime and target
        mask_corr_gt = create_border_mask(flow_gt)
        # if mask_gt has too little valid areas, overwrite to use that mask in anycase
        if mask_corr_gt.sum() < mask_corr_gt.shape[-1] * mask_corr_gt.shape[-2] * self.min_fraction_valid_corr:
            mask = mask_corr_gt
        return image_prime.squeeze(0), flow_gt.squeeze(0), mask.squeeze(0)


class CompositeFlow(nn.Module):

    def __init__(self,
                 apply_keys='all',
                 include_transforms=['hom', 'affine'],
                 random_alpha=0.065,
                 random_s=0.6,
                 random_tx=0.3,
                 random_ty=0.1,
                 random_t_tps=0,
                 random_t_hom=0.3,
                 random_t_tps_for_afftps=0,
                 parameterize_with_gaussian=False,
                 min_fraction_valid_corr=0.1,
                 add_elastic=False,
                 ):
        super().__init__()
        self.apply_keys = apply_keys
        self.transforms = nn.ModuleList()
        for t in include_transforms:
            if t == 'hom':
                self.transforms.append(RandomHomography(apply_keys, random_t_hom=random_t_hom,
                                       parameterize_with_gaussian=parameterize_with_gaussian, return_flow=True))
            elif t == 'affine':
                self.transforms.append(RandomAffine(apply_keys, random_alpha=random_alpha, random_s=random_s, random_tx=random_tx,
                                       random_ty=random_ty, parameterize_with_gaussian=parameterize_with_gaussian, return_flow=True))
            elif t == 'tps':
                self.transforms.append(RandomTPS(apply_keys, random_t_tps=random_t_tps,
                                       parameterize_with_gaussian=parameterize_with_gaussian, return_flow=True))
            elif t == 'afftps':
                self.transforms.append(RandomAffineTPS(apply_keys, random_alpha=random_alpha, random_s=random_s, random_tx=random_tx, random_ty=random_ty,
                                       random_t_tps_for_afftps=random_t_tps_for_afftps, parameterize_with_gaussian=parameterize_with_gaussian, return_flow=True))
        self.add_elastic = add_elastic
        if self.add_elastic:
            self.elastic_trafo = RandomElastic(apply_keys, return_flow=True)
        self.min_fraction_valid_corr = min_fraction_valid_corr

    def forward(self, sample):
        trafo = random.choice(self.transforms)
        flow = trafo(sample)
        if self.add_elastic:
            flow = self.elastic_trafo(sample, flow=flow)

        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys
        for key in apply_keys:
            val = sample[key]
            if key == 'image_prime':
                sample[key], sample[key + '_flow'], sample[key +
                                                           '_mask'] = self.apply_transform(val, flow)
            else:
                raise ValueError
        return sample

    def apply_transform(self, image, flow_gt):
        assert image.shape[-2:] == flow_gt.shape[-2:]
        if image.ndim == 3:
            image.unsqueeze_(0)
        image_prime, mask = warp(
            image, flow_gt, padding_mode='zeros', return_mask=True)
        # ground truth correspondence mask for flow between target prime and target
        mask_corr_gt = create_border_mask(flow_gt)
        # if mask_gt has too little valid areas, overwrite to use that mask in anycase
        if mask_corr_gt.sum() < mask_corr_gt.shape[-1] * mask_corr_gt.shape[-2] * self.min_fraction_valid_corr:
            mask = mask_corr_gt
        return image_prime.squeeze(0), flow_gt.squeeze(0), mask.squeeze(0)


class CenterCrop(torchvision.transforms.CenterCrop):

    def __init__(self, apply_keys='all', **kwargs):
        super().__init__(**kwargs)
        self.apply_keys = apply_keys

    def forward(self, sample):
        if self.apply_keys == 'all':
            apply_keys = list(sample)
        else:
            apply_keys = self.apply_keys
        w, h = get_image_size(sample, apply_keys)
        for key in apply_keys:
            val = sample[key]
            if key in ['image',
                       'image_ref',
                       'image_prime',
                       'semantic',
                       'image_prime_flow',
                       'image_prime_mask']:
                sample[key] = super().forward(val)
            elif key == 'corr_pts':
                pts1 = sample['corr_pts_ref']
                pts2 = sample['corr_pts']
                pts1, pts2 = self.adjust_correspondences(pts1, pts2, h, w)
                sample['corr_pts_ref'] = pts1
                sample['corr_pts'] = pts2
            elif key in ['filename', 'image_prime_idx', 'corr_pts_ref']:
                pass
            else:
                raise ValueError
        return sample

    def adjust_correspondences(self, pts1, pts2, image_height, image_width):
        crop_height, crop_width = self.size
        if crop_width > image_width or crop_height > image_height:
            raise NotImplementedError
        top = int(round((image_height - crop_height) / 2.0))
        left = int(round((image_width - crop_width) / 2.0))
        pts1[:, 0] = pts1[:, 0] - left
        pts1[:, 1] = pts1[:, 1] - top
        pts2[:, 0] = pts2[:, 0] - left
        pts2[:, 1] = pts2[:, 1] - top

        # remove all correspondence not seen in cropped image
        in_im = ((torch.round(pts1[:, 0]) >= 0) & (torch.round(pts1[:, 0]) < crop_width) &
                 (torch.round(pts2[:, 0]) >= 0) & (torch.round(pts2[:, 0]) < crop_width) &
                 (torch.round(pts1[:, 1]) >= 0) & (torch.round(pts1[:, 1]) < crop_height) &
                 (torch.round(pts2[:, 1]) >= 0) & (torch.round(pts2[:, 1]) < crop_height))
        pts1 = pts1[in_im]
        pts2 = pts2[in_im]
        return pts1, pts2
