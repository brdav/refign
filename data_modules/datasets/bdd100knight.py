import os
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from PIL import Image

from ..transforms import pillow_interp_codes


class BDD100kNight(torch.utils.data.Dataset):

    orig_dims = (720, 1280)

    def __init__(
            self,
            root: str,
            stage: str = "test",
            load_keys: Union[List[str], str] = ["image", "semantic"],
            dims: Union[List[int], Tuple[int, int]] = (720, 1280),
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

        self.paths = {k: [] for k in self.load_keys}

        img_paths = [i_id.strip() for i_id in open(os.path.join(os.path.dirname(
            __file__), 'lists/images_trainval_night_correct_filenames.txt'))]
        for img_path in img_paths:
            _, _, split, name = img_path.split('/')
            self.paths['image'].append(os.path.join(
                self.root, 'images', '10k', split, name))
            self.paths['semantic'].append(os.path.join(
                self.root, 'labels', 'sem_seg', 'masks', split, name.replace('.jpg', '.png')))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item.
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
