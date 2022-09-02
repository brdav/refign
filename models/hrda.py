import random
from functools import wraps
from typing import Callable

import torch
import torch.nn as nn


def extract_crop(img, crop_size, divisible=1):
    """Randomly get a crop bounding box."""
    img_h, img_w = img.shape[-2:]
    assert crop_size[0] > 0 and crop_size[1] > 0
    if img_h == crop_size[-2] and img_w == crop_size[-1]:
        return (0, img_h, 0, img_w)
    margin_h = max(img_h - crop_size[-2], 0)
    margin_w = max(img_w - crop_size[-1], 0)
    offset_h = random.randrange(
        0, (margin_h + 1) // divisible) * divisible
    offset_w = random.randrange(
        0, (margin_w + 1) // divisible) * divisible
    crop_y1, crop_y2 = int(offset_h), int(offset_h + crop_size[0])
    crop_x1, crop_x2 = int(offset_w), int(offset_w + crop_size[1])

    crop_boxes = [[crop_y1, crop_y2, crop_x1, crop_x2]]
    crop_imgs = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]

    return crop_imgs, crop_boxes


def hr_crop_slice(crop_box, scale):
    crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(crop_box, scale)
    return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)


def scale_box(box, scale):
    y1, y2, x1, x2 = box
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2


def extract_slide_crop(img, crop_size):
    h_stride, w_stride = [e // 2 for e in crop_size]
    h_crop, w_crop = crop_size
    h_img, w_img = img.shape[-2:]
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    crop_imgs, crop_boxes = [], []
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_imgs.append(img[:, :, y1:y2, x1:x2])
            crop_boxes.append([y1, y2, x1, x2])
    crop_imgs = torch.cat(crop_imgs, dim=0)
    # shape: feature levels, crops * batch size x c x h x w
    return crop_imgs, crop_boxes


def hrda_backbone(self, head_os: int, is_teacher: bool = False) -> Callable:
    def hrda_backbone_decorator(fn: Callable) -> Callable:

        @wraps(fn)
        def inner_fn(*args, **kwargs):

            x = args[0]

            if self.training and not is_teacher:
                hrda_crop_coord_divisible = head_os * 2.0

                # lr input
                lr_x = nn.functional.interpolate(
                    x, scale_factor=0.5, mode='bilinear', align_corners=False)  # TODO: try anti alias

                # hr input
                crop_size = lr_x.shape[-2:]
                hr_x, hr_boxes = extract_crop(
                    x, crop_size, hrda_crop_coord_divisible)

            else:
                # lr input
                lr_x = nn.functional.interpolate(
                    x, scale_factor=0.5, mode='bilinear', align_corners=False)  # TODO: try anti alias

                # hr input
                crop_size = lr_x.shape[-2:]
                hr_x, hr_boxes = extract_slide_crop(x, crop_size)

            # forward
            lr_bs = lr_x.shape[0]
            hr_bs = hr_x.shape[0]
            both_x = torch.cat((lr_x, hr_x))
            both_feats = fn(both_x, *args[1:], **kwargs)
            lr_feats, hr_feats = zip(
                *(torch.split(f, [lr_bs, hr_bs]) for f in both_feats))
            return lr_feats, hr_feats, hr_boxes

        return inner_fn
    return hrda_backbone_decorator


def hrda_head(self, hrda_scale_attention: nn.Module, head_os: int, is_teacher: bool = False) -> Callable:
    def hrda_head_decorator(fn: Callable) -> Callable:

        @wraps(fn)
        def inner_fn(*args, **kwargs):

            lr_feats, hr_feats, hr_boxes = args[0]

            if self.training and not is_teacher:

                att = torch.sigmoid(hrda_scale_attention(lr_feats))
                crop_box = hr_boxes[0]
                crop_size = (crop_box[1] - crop_box[0],
                             crop_box[3] - crop_box[2])

                # forward
                lr_bs = lr_feats[0].shape[0]
                hr_bs = hr_feats[0].shape[0]
                both_feats = [torch.cat(el) for el in zip(lr_feats, hr_feats)]
                both_seg = fn(both_feats, *args[1:], **kwargs)
                lr_seg, hr_seg = torch.split(both_seg, [lr_bs, hr_bs])

                # lr
                mask = lr_seg.new_zeros(
                    [lr_seg.shape[0], 1, *lr_seg.shape[2:]])
                sc_os = 2.0 * head_os
                slc = hr_crop_slice(crop_box, sc_os)
                mask[:, :, slc[0], slc[1]] = 1
                att = att * mask
                lr_seg = (1 - att) * lr_seg
                up_lr_seg = nn.functional.interpolate(
                    lr_seg, scale_factor=2, mode='bilinear', align_corners=False)

                # hr
                up_att = nn.functional.interpolate(
                    att, scale_factor=2, mode='bilinear', align_corners=False)
                hr_seg_inserted = torch.zeros_like(up_lr_seg)
                slc = hr_crop_slice(crop_box, head_os)
                hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg

                # upsample hr_logits to crop_size
                hr_logits = nn.functional.interpolate(
                    hr_seg, crop_size, mode='bilinear', align_corners=False)

                logits = up_att * hr_seg_inserted + up_lr_seg

                return logits, hr_logits, crop_box

            else:

                att = torch.sigmoid(hrda_scale_attention(lr_feats))

                # forward
                lr_bs = lr_feats[0].shape[0]
                hr_bs = hr_feats[0].shape[0]
                both_feats = [torch.cat(a) for a in zip(lr_feats, hr_feats)]
                both_seg = fn(both_feats, *args[1:], **kwargs)
                lr_seg, crop_seg_logits = torch.split(both_seg, [lr_bs, hr_bs])

                # lr
                lr_seg = (1 - att) * lr_seg
                up_lr_seg = nn.functional.interpolate(
                    lr_seg, scale_factor=2, mode='bilinear', align_corners=False)

                # hr
                dev = hr_feats[0][0].device
                bs = lr_seg.shape[0]
                h_img, w_img = 0, 0
                for i in range(len(hr_boxes)):
                    hr_boxes[i] = scale_box(hr_boxes[i], head_os)
                    y1, y2, x1, x2 = hr_boxes[i]
                    if h_img < y2:
                        h_img = y2
                    if w_img < x2:
                        w_img = x2
                preds = torch.zeros((bs, self.num_classes, h_img, w_img),
                                    device=dev)
                count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)
                for i in range(len(hr_boxes)):
                    y1, y2, x1, x2 = hr_boxes[i]
                    crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                    preds += nn.functional.pad(crop_seg_logit,
                                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                                int(preds.shape[2] - y2)))

                    count_mat[:, :, y1:y2, x1:x2] += 1
                assert (count_mat == 0).sum() == 0
                hr_seg = preds / count_mat

                up_att = nn.functional.interpolate(
                    att, scale_factor=2, mode='bilinear', align_corners=False)
                logits = up_att * hr_seg + up_lr_seg

                return logits

        return inner_fn
    return hrda_head_decorator
