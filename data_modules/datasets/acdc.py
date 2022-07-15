import os
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from PIL import Image

from ..transforms import pillow_interp_codes


class ACDC(torch.utils.data.Dataset):

    orig_dims = (1080, 1920)

    def __init__(
            self,
            root: str,
            stage: str = "train",
            condition: Union[List[str], str] = [
                "fog", "night", "rain", "snow"],
            load_keys: Union[List[str], str] = [
                "image_ref", "image", "semantic"],
            dims: Union[Tuple[int, int], List[int]] = (1080, 1920),
            transforms: Optional[Callable] = None,
            predict_on: Optional[str] = None,
            **kwargs
    ) -> None:
        super().__init__()
        self.root = root
        self.dims = dims
        self.transforms = transforms

        assert stage in ["train", "val", "test", "predict"]
        self.stage = stage

        # mapping from stage to splits
        if self.stage == 'train':
            self.split = 'train'
        elif self.stage == 'val':
            self.split = 'val'
        elif self.stage == 'test':
            self.split = 'val'  # test on val split
        elif self.stage == 'predict':
            if not predict_on:
                self.split = 'test'  # predict on test split
            else:
                self.split = predict_on

        if isinstance(condition, str):
            self.condition = [condition]
        else:
            self.condition = condition

        if isinstance(load_keys, str):
            self.load_keys = [load_keys]
        else:
            self.load_keys = load_keys

        self.paths = {k: []
                      for k in ['image', 'image_ref', 'semantic']}

        self.images_dir = os.path.join(self.root, 'rgb_anon')
        self.semantic_dir = os.path.join(self.root, 'gt')
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.semantic_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "condition" are inside the "root" directory')

        for cond in self.condition:
            img_parent_dir = os.path.join(self.images_dir, cond, self.split)
            semantic_parent_dir = os.path.join(
                self.semantic_dir, cond, self.split)
            for recording in os.listdir(img_parent_dir):
                img_dir = os.path.join(img_parent_dir, recording)
                semantic_dir = os.path.join(semantic_parent_dir, recording)
                for file_name in os.listdir(img_dir):
                    for k in ['image', 'image_ref', 'semantic']:
                        if k == 'image':
                            file_path = os.path.join(img_dir, file_name)
                        elif k == 'image_ref':
                            ref_img_dir = img_dir.replace(
                                self.split, self.split + '_ref')
                            ref_file_name = file_name.replace(
                                'rgb_anon', 'rgb_ref_anon')
                            file_path = os.path.join(
                                ref_img_dir, ref_file_name)
                        elif k == 'semantic':
                            semantic_file_name = file_name.replace(
                                'rgb_anon.png', 'gt_labelTrainIds.png')
                            file_path = os.path.join(
                                semantic_dir, semantic_file_name)
                        self.paths[k].append(file_path)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        sample: Any = {}
        sample['filename'] = self.paths['image'][index].split('/')[-1]

        for k in self.load_keys:
            if k in ['image', 'image_ref']:
                data = Image.open(self.paths[k][index]).convert('RGB')
                if data.size != self.dims[::-1]:
                    data = data.resize(
                        self.dims[::-1], resample=pillow_interp_codes['bilinear'])
            elif k == 'semantic':
                data = Image.open(self.paths[k][index])
                if data.size != self.dims[::-1]:
                    data = data.resize(
                        self.dims[::-1], resample=pillow_interp_codes['nearest'])
            else:
                raise ValueError('invalid load_key')
            sample[k] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return len(next(iter(self.paths.values())))
