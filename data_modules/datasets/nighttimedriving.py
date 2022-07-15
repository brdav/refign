import os
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from PIL import Image

from ..transforms import pillow_interp_codes


class NighttimeDriving(torch.utils.data.Dataset):

    orig_dims = (1080, 1920)

    def __init__(
            self,
            root: str,
            stage: str = "test",
            load_keys: Union[List[str], str] = ["image", "semantic"],
            dims: Union[Tuple[int, int], List[int]] = (1080, 1920),
            transforms: Optional[Callable] = None,
            **kwargs
    ) -> None:
        super().__init__()
        self.root = root
        self.dims = dims
        self.transforms = transforms

        assert stage == 'test'
        self.stage = stage
        self.split = 'test'

        if isinstance(load_keys, str):
            self.load_keys = [load_keys]
        else:
            self.load_keys = load_keys

        self.paths = {k: [] for k in ['image', 'semantic']}

        self.images_dir = os.path.join(self.root, 'leftImg8bit')
        self.semantic_dir = os.path.join(
            self.root, 'gtCoarse_daytime_trainvaltest')
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.semantic_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "condition" are inside the "root" directory')

        img_dir = os.path.join(self.images_dir, self.split, 'night')
        semantic_dir = os.path.join(self.semantic_dir, self.split, 'night')
        for file_name in os.listdir(img_dir):
            for k in ['image', 'semantic']:
                if k == 'image':
                    file_path = os.path.join(img_dir, file_name)
                elif k == 'semantic':
                    semantic_file_name = file_name.replace(
                        'leftImg8bit.png', 'gtCoarse_labelTrainIds.png')
                    file_path = os.path.join(semantic_dir, semantic_file_name)
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
            if k == 'image':
                data = Image.open(self.paths[k][index]).convert('RGB')
                if data.size != self.dims[::-1]:
                    data = data.resize(
                        self.dims[::-1], resample=pillow_interp_codes['bilinear'])
            elif k == 'semantic':
                data = Image.open(self.paths[k][index])
                if data.size != self.dims[::-1]:
                    data = data.resize(
                        self.dims[::-1], resample=pillow_interp_codes['nearest'])
            sample[k] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return len(next(iter(self.paths.values())))
