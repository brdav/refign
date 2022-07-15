import json
import os
import random
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from PIL import Image

from ..transforms import pillow_interp_codes


class Cityscapes(torch.utils.data.Dataset):

    orig_dims = (1024, 2048)

    def __init__(
            self,
            root: str,
            stage: str = "train",
            load_keys: Union[List[str], str] = ["image", "semantic"],
            dims: Union[List[int], Tuple[int, int]] = (1024, 2048),
            transforms: Optional[Callable] = None,
            rcs_enabled: bool = False,
            rcs_class_temp: float = 0.01,
            rcs_min_crop_ratio: float = 0.5,
            rcs_min_pixels: int = 3000,
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
            self.split = 'test'  # predict on test split

        if isinstance(load_keys, str):
            self.load_keys = [load_keys]
        else:
            self.load_keys = load_keys

        self.paths = {k: [] for k in self.load_keys}

        self.images_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        self.semantic_dir = os.path.join(self.root, 'gtFine', self.split)
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.semantic_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            semantic_dir = os.path.join(self.semantic_dir, city)
            for file_name in os.listdir(img_dir):
                for k in self.load_keys:
                    if k == 'image':
                        file_path = os.path.join(img_dir, file_name)
                    elif k == 'semantic':
                        semantic_file_name = file_name.replace(
                            'leftImg8bit.png', 'gtFine_labelTrainIds.png')
                        file_path = os.path.join(
                            semantic_dir, semantic_file_name)
                    self.paths[k].append(file_path)

        self.rcs_enabled = rcs_enabled
        self.rcs_class_temp = rcs_class_temp
        self.rcs_min_crop_ratio = rcs_min_crop_ratio
        self.rcs_min_pixels = rcs_min_pixels
        if self.rcs_enabled:

            self.rcs_classes, self.rcs_classprob = self.get_rcs_class_probs(
                self.root, self.rcs_class_temp)
            # print(f'RCS Classes: {self.rcs_classes}')
            # print(f'RCS ClassProb: {self.rcs_classprob}')

            with open(os.path.join(self.root, 'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.indices_with_class = {}
            for c in self.rcs_classes:
                self.indices_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.indices_with_class[c].append(
                            self.paths['semantic'].index(os.path.expandvars(file)))
                assert len(self.indices_with_class[c]) > 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item.
        """
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            return self.load_and_augment_sample(index)

    def load_and_augment_sample(self, index):
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
            else:
                raise ValueError('invalid load_key')
            sample[k] = data

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self) -> int:
        return len(next(iter(self.paths.values())))

    def get_rare_class_sample(self):
        assert 'image' in self.load_keys and 'semantic' in self.load_keys

        c = random.choices(
            self.rcs_classes, weights=self.rcs_classprob, k=1)[0]
        index = random.choice(self.indices_with_class[c])

        sample = self.load_and_augment_sample(index)
        if self.rcs_min_crop_ratio > 0:
            for _ in range(10):
                n_class = torch.sum(sample['semantic'].data == c)
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                # Sample a new random crop from source image i1.
                # Please note, that self.source.__getitem__(idx) applies the
                # preprocessing pipeline to the loaded image, which includes
                # RandomCrop, and results in a new crop of the image.
                sample = self.load_and_augment_sample(index)

        return sample

    @staticmethod
    def get_rcs_class_probs(data_root, temperature):
        with open(os.path.join(data_root, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)
        overall_class_stats = {}
        for s in sample_class_stats:
            s.pop('file')
            for c, n in s.items():
                c = int(c)
                if c not in overall_class_stats:
                    overall_class_stats[c] = n
                else:
                    overall_class_stats[c] += n
        overall_class_stats = {
            k: v
            for k, v in sorted(
                overall_class_stats.items(), key=lambda item: item[1])
        }
        freq = torch.tensor(list(overall_class_stats.values()))
        freq = freq / torch.sum(freq)
        freq = 1 - freq
        freq = torch.softmax(freq / temperature, dim=-1)

        return list(overall_class_stats.keys()), freq.numpy()
