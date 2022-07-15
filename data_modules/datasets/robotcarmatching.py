import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..transforms import pillow_interp_codes


class RobotCarMatching(torch.utils.data.Dataset):

    def __init__(
            self,
            root: str,
            stage: str = "test",
            load_keys: Union[List[str], str] = ["image_ref", "image"],
            dims: Union[List[int], Tuple[int, int]] = (1024, 1024),
            transforms: Optional[Callable] = None,
            resize_filter: str = 'lanczos',
            **kwargs
    ) -> None:
        super().__init__()
        self.root = root
        self.dims = dims
        self.transforms = transforms
        self.resize_filter = resize_filter

        assert stage in ['test', 'predict']
        self.stage = stage

        if isinstance(load_keys, str):
            self.load_keys = [load_keys]
        else:
            self.load_keys = load_keys

        if not os.path.isdir(self.root):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" are inside the "root" directory')

        # loads around 5 GB into RAM
        self.df = pd.read_csv(os.path.join(
            self.root, 'test6511.csv'), dtype=str)
        self.images_dir = os.path.join(self.root, 'images')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item.
        """
        sample: Any = {}

        scene = self.df['scene'][index]
        if scene == '/':
            scene = '.'
        src_name = os.path.join(scene, self.df['source_image'][index])
        trg_name = os.path.join(scene, self.df['target_image'][index])

        # target points
        pts_trg_x = np.array(
            list(map(float, self.df['XB'][index].split(';')))).astype(np.float32)
        pts_trg_y = np.array(
            list(map(float, self.df['YB'][index].split(';')))).astype(np.float32)
        pts_trg = np.stack((pts_trg_x, pts_trg_y), axis=1)
        # source points
        pts_src_x = np.array(
            list(map(float, self.df['XA'][index].split(';')))).astype(np.float32)
        pts_src_y = np.array(
            list(map(float, self.df['YA'][index].split(';')))).astype(np.float32)
        pts_src = np.stack((pts_src_x, pts_src_y), axis=1)

        # in the end, we test 'trg_src_flow', so the warp of src into trg
        for k in self.load_keys:
            if k == 'image_ref':
                data = Image.open(os.path.join(
                    self.images_dir, src_name)).convert('RGB')
                orig_im_size = data.size
                if self.dims is not None and data.size != self.dims[::-1]:
                    data = data.resize(
                        self.dims[::-1], resample=pillow_interp_codes[self.resize_filter])
                    x_scale = self.dims[1] / float(orig_im_size[0])
                    y_scale = self.dims[0] / float(orig_im_size[1])
                    pts_src[:, 0] = x_scale * pts_src[:, 0]
                    pts_src[:, 1] = y_scale * pts_src[:, 1]
            elif k == 'image':
                data = Image.open(os.path.join(
                    self.images_dir, trg_name)).convert('RGB')
                orig_im_size = data.size
                if self.dims is not None and data.size != self.dims[::-1]:
                    data = data.resize(
                        self.dims[::-1], resample=pillow_interp_codes[self.resize_filter])
                    x_scale = self.dims[1] / float(orig_im_size[0])
                    y_scale = self.dims[0] / float(orig_im_size[1])
                    pts_trg[:, 0] = x_scale * pts_trg[:, 0]
                    pts_trg[:, 1] = y_scale * pts_trg[:, 1]
            else:
                raise ValueError('invalid load_key')
            sample[k] = data

        sample['corr_pts'] = torch.from_numpy(pts_trg)  # r x 2
        sample['corr_pts_ref'] = torch.from_numpy(pts_src)  # r x 2

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return len(self.df)
