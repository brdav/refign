import copy
import os
import random
import time
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ..transforms import pillow_interp_codes


class MegaDepth(torch.utils.data.Dataset):
    """ MegaDepth dataset. Retrieves either pairs of matching images and their corresponding ground-truth flow
    (that is actually sparse) or single images. 
    """

    cfg = {
        'train_split': 'lists/train_scenes_MegaDepth.txt',
        'train_debug_split': 'lists/train_debug_scenes_MegaDepth.txt',
        'val_split': 'lists/validation_scenes_MegaDepth.txt',
        'test_split': 'lists/test_scenes_MegaDepth.txt',
        'train_debug_num_per_scene': 10,
        'train_num_per_scene': 300,
        'val_num_per_scene': 25,
        'min_overlap_ratio': 0.3,
        'max_overlap_ratio': 1.,
    }

    def __init__(self,
                 root: str,
                 stage: str = 'train',
                 load_keys: Union[List[str], str] = [
                     "image_ref", "image", "image_prime"],
                 dims: Optional[Union[Tuple[int, int], List[int]]] = None,
                 transforms: Optional[Callable] = None,
                 exchange_images_with_proba: float = 0,
                 store_scene_info_in_memory: bool = False,
                 debug: bool = False):
        """
        Args:
            root: root directory
            cfg: config (dictionary)
            split: 'train' or 'val'
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            co_transform: transforms to apply to both image pairs and corresponding flow field
            compute_mask_zero_borders: output mask of zero borders ?
            store_scene_info_in_memory: store all scene info in cpu memory? requires at least 50GB for training but
                                        sampling at each epoch is faster.
        Output in __getitem__:
            if self.two_views:
                source_image
                target_image
                flow_map: flow fields in flow coordinate system, relating flow to source image
                correspondence_mask: visible and valid correspondences
                source_image_size
                sparse: True
                if mask_zero_borders:
                    mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise
            else:
                image
                scene: id of the scene
        """

        super().__init__()
        self.root = root
        self.dims = dims
        self.transforms = transforms
        self.exchange_images_with_proba = exchange_images_with_proba

        assert stage in ["train", "val", "test", "predict"]
        self.stage = stage

        # mapping from stage to splits
        if self.stage == 'train':
            self.split = 'train'
        elif self.stage == 'val':
            self.split = 'val'
        elif self.stage == 'test':
            self.split = 'test'
        elif self.stage == 'predict':
            self.split = 'test'  # predict on test split

        if isinstance(load_keys, str):
            self.load_keys = [load_keys]
        else:
            self.load_keys = load_keys

        if debug:  # ignore actual split in debug mode
            self.split = 'train_debug'

        if self.split == 'test':
            assert not 'image_prime' in self.load_keys
            assert self.exchange_images_with_proba == 0
            import pandas as pd
            self.df = pd.read_csv(os.path.join(
                self.root, 'Test', 'test1600Pairs.csv'), dtype=str)
            self.images_dir = os.path.join(self.root, 'Test', 'test1600Pairs')

        else:
            self.scene_info_path = os.path.join(self.root, 'scene_info')
            with open(os.path.join(os.path.dirname(__file__), self.cfg[self.split + '_split']), 'r') as f:
                self.scenes = f.read().split()

            if 'image_ref' in self.load_keys and 'image' in self.load_keys:
                self.two_views = True
            else:
                self.two_views = False
            if 'image_prime' in self.load_keys:
                assert self.two_views
            if 'image' in self.load_keys:
                assert self.two_views

            self.items = []
            if not self.two_views:
                # if single view, always just store
                self.store_scene_info_in_memory = True
            else:
                self.store_scene_info_in_memory = store_scene_info_in_memory

            if self.store_scene_info_in_memory:
                # it will take around 35GB, you need at least 50GB of cpu memory to train
                self.save_scene_info()
            self.sample_new_items()
        print('MegaDepth: {} dataset comprises {} image pairs'.format(
            self.split, self.__len__()))

    def save_scene_info(self):
        print('\nMegaDepth {}: Storing info about scenes on memory...\nThis will take some time'.format(
            self.split))
        start = time.time()
        self.images = {}
        if self.two_views:
            self.points3D_id_to_2D = {}
            self.pairs = {}

        # for scene in tqdm(self.scenes):
        for i, scene in enumerate(self.scenes):
            path = os.path.join(self.scene_info_path, '%s.0.npz' % scene)
            if not os.path.exists(path):
                print(f'Scene {scene} does not have an info file')
                continue
            info = np.load(path, allow_pickle=True)

            valid = ((info['image_paths'] != None) &
                     (info['depth_paths'] != None))
            self.images[scene] = info['image_paths'][valid].copy()

            if self.two_views:
                self.points3D_id_to_2D[scene] = info['points3D_id_to_2D'][valid].copy(
                )

                # write pairs that have a correct overlap ratio
                # N_img x N_img where N_img is len(self.images[scene])
                mat = info['overlap_matrix'][valid][:, valid]
                pairs = (
                    (mat > self.cfg['min_overlap_ratio'])
                    & (mat <= self.cfg['max_overlap_ratio']))
                pairs = np.stack(np.where(pairs), -1)
                self.pairs[scene] = [(i, j, mat[i, j]) for i, j in pairs]

            del info
        total = time.time() - start
        print('Storing took {} s'.format(total))

    def sample_new_items(self, seed=400):
        print('\nMegaDepth {}: Sampling new images or pairs with seed {}. \nThis will take some time...'
              .format(self.split, seed))
        start_time = time.time()
        self.items = []

        num = self.cfg[self.split + '_num_per_scene']

        for i, scene in enumerate(self.scenes):
            path = os.path.join(self.scene_info_path, '%s.0.npz' % scene)
            if not os.path.exists(path):
                print(f'Scene {scene} does not have an info file')
                continue
            if self.two_views and self.store_scene_info_in_memory:
                # sampling is just accessing the pairs
                pairs = np.array(self.pairs[scene])
                if len(pairs) > num:
                    selected = np.random.RandomState(seed).choice(
                        len(pairs), num, replace=False)
                    pairs = pairs[selected]

                pairs_one_direction = [(scene, int(i), int(j), k)
                                       for i, j, k in pairs]
                self.items.extend(pairs_one_direction)
            elif self.two_views:
                # sample all infos from the scenes
                info = np.load(path, allow_pickle=True, mmap_mode='r')
                valid = ((info['image_paths'] != None) &
                         (info['depth_paths'] != None))
                paths = info['image_paths'][valid]

                points3D_id_to_2D = info['points3D_id_to_2D'][valid]

                mat = info['overlap_matrix'][valid][:, valid]
                pairs = (
                        (mat > self.cfg['min_overlap_ratio'])
                    & (mat <= self.cfg['max_overlap_ratio']))
                pairs = np.stack(np.where(pairs), -1)
                if len(pairs) > num:
                    selected = np.random.RandomState(seed).choice(
                        len(pairs), num, replace=False)
                    pairs = pairs[selected]

                for pair_idx in range(len(pairs)):
                    idx1 = pairs[pair_idx, 0]
                    idx2 = pairs[pair_idx, 1]
                    matches = np.array(
                        list(points3D_id_to_2D[idx1].keys() & points3D_id_to_2D[idx2].keys()))

                    point2D1 = [np.array(points3D_id_to_2D[idx1][match], dtype=np.float32).reshape(1, 2) for
                                match in matches]

                    point2D2 = [np.array(points3D_id_to_2D[idx2][match], dtype=np.float32).reshape(1, 2) for
                                match in matches]

                    image_pair_bundle = {
                        'image_path1': paths[idx1],
                        'image_path2': paths[idx2],
                        '2d_matches_1': point2D1.copy(),
                        '2d_matches_2': point2D2.copy()
                    }
                    self.items.append(image_pair_bundle)
            else:
                # single view, just sample new paths to imahes
                ids = np.arange(len(self.images[scene]))
                if len(ids) > num:
                    ids = np.random.RandomState(seed).choice(
                        ids, num, replace=False)
                ids = [(scene, i) for i in ids]
                self.items.extend(ids)

        if 'debug' in self.split:
            orig_copy = copy.deepcopy(self.items)
            for _ in range(10):
                self.items = self.items + copy.deepcopy(orig_copy)

        np.random.RandomState(seed).shuffle(self.items)
        end_time = time.time() - start_time
        print('Sampling took {} s. Sampled {} items'.format(
            end_time, len(self.items)))

    def __len__(self):
        if self.split == 'train':
            return 30000
        elif self.split == 'test':
            return len(self.df)
        else:
            return len(self.items)

    def _read_pair_info(self, scene, idx1, idx2):
        # when scene info are stored in memory
        matches = np.array(list(self.points3D_id_to_2D[scene][idx1].keys(
        ) & self.points3D_id_to_2D[scene][idx2].keys()))

        # obtain 2D coordinate for all matches between the pair
        point2D1 = [np.array(self.points3D_id_to_2D[scene][idx1][match], dtype=np.float32).reshape(1, 2) for
                    match in matches]
        point2D2 = [np.array(self.points3D_id_to_2D[scene][idx2][match], dtype=np.float32).reshape(1, 2) for
                    match in matches]

        image_pair_bundle = {
            'image_path1': self.images[scene][idx1],
            'image_path2': self.images[scene][idx2],
            '2d_matches_1': point2D1,
            '2d_matches_2': point2D2
        }
        return image_pair_bundle

    def _read_single_view(self, scene, idx):
        path = os.path.join(self.root, self.images[scene][idx])
        image = Image.open(path).convert('RGB')

        if self.dims is not None and image.size != self.dims[::-1]:
            # resize to a fixed load_size and rescale the keypoints accordingly
            image = image.resize(
                self.dims[::-1], resample=pillow_interp_codes['lanczos'])

        return image

    def recover_pair(self, pair_metadata, exchange_images=False):
        if exchange_images:
            image_path1 = os.path.join(self.root, pair_metadata['image_path2'])
            image_path2 = os.path.join(self.root, pair_metadata['image_path1'])
            points2D1_from_file = np.concatenate(
                pair_metadata['2d_matches_2'], axis=0)  # Nx2
            points2D2_from_file = np.concatenate(
                pair_metadata['2d_matches_1'], axis=0)  # Nx2
        else:
            image_path1 = os.path.join(self.root, pair_metadata['image_path1'])
            image_path2 = os.path.join(self.root, pair_metadata['image_path2'])
            points2D1_from_file = np.concatenate(
                pair_metadata['2d_matches_1'], axis=0)  # Nx2
            points2D2_from_file = np.concatenate(
                pair_metadata['2d_matches_2'], axis=0)  # Nx2

        image1 = Image.open(image_path1).convert('RGB')
        image2 = Image.open(image_path2).convert('RGB')

        if self.dims is not None and image1.size != self.dims[::-1]:
            # resize to a fixed load_size and rescale the keypoints accordingly
            w1, h1 = image1.size
            image1 = image1.resize(
                self.dims[::-1], resample=pillow_interp_codes['lanczos'])
            points2D1_from_file[:, 0] *= float(self.dims[1]) / float(w1)
            points2D1_from_file[:, 1] *= float(self.dims[0]) / float(h1)
        if self.dims is not None and image2.size != self.dims[::-1]:
            # resize to a fixed load_size and rescale the keypoints accordingly
            w2, h2 = image2.size
            image2 = image2.resize(
                self.dims[::-1], resample=pillow_interp_codes['lanczos'])
            points2D2_from_file[:, 0] *= float(self.dims[1]) / float(w2)
            points2D2_from_file[:, 1] *= float(self.dims[0]) / float(h2)

        points2D1_from_file = torch.from_numpy(points2D1_from_file)
        points2D2_from_file = torch.from_numpy(points2D2_from_file)

        return image1, image2, points2D1_from_file, points2D2_from_file

    def __getitem__(self, index):
        """
        Args:
            idx
        Returns: Dictionary with fieldnames:
            if self.two_views:
                source_image
                target_image
                flow_map: flow fields in target coordinate system, relating target to source image
                correspondence_mask: visible and valid correspondences
                source_image_size
                sparse: True
                if mask_zero_borders:
                    mask_zero_borders: bool tensor equal to 1 where the target image is not equal to 0, 0 otherwise
            else:
                image
                scene: id of the scene
        """

        if self.split == 'test':
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

            for k in self.load_keys:
                if k == 'image_ref':
                    data = Image.open(os.path.join(
                        self.images_dir, src_name)).convert('RGB')
                    orig_im_size = data.size
                    if self.dims is not None and data.size != self.dims[::-1]:
                        data = data.resize(
                            self.dims[::-1], resample=pillow_interp_codes['lanczos'])
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
                            self.dims[::-1], resample=pillow_interp_codes['lanczos'])
                        x_scale = self.dims[1] / float(orig_im_size[0])
                        y_scale = self.dims[0] / float(orig_im_size[1])
                        pts_trg[:, 0] = x_scale * pts_trg[:, 0]
                        pts_trg[:, 1] = y_scale * pts_trg[:, 1]
                sample[k] = data

            sample['corr_pts'] = torch.from_numpy(pts_trg)  # r x 2
            sample['corr_pts_ref'] = torch.from_numpy(pts_src)  # r x 2

        else:
            if self.two_views:
                if self.store_scene_info_in_memory:
                    scene, idx1, idx2, overlap = self.items[index]
                    pair_metadata = self._read_pair_info(scene, idx1, idx2)
                else:
                    pair_metadata = self.items[index]
                if random.random() < self.exchange_images_with_proba:
                    source, target, pts_source, pts_target = self.recover_pair(
                        pair_metadata, exchange_images=True)
                else:
                    source, target, pts_source, pts_target = self.recover_pair(
                        pair_metadata, exchange_images=False)
                sample = {
                    "image_ref": source,
                    "image": target,
                    "corr_pts_ref": pts_source,
                    "corr_pts": pts_target,
                    # for consistency
                    "image_prime_idx": torch.ones(1, dtype=torch.long)
                }

                if 'image_prime' in self.load_keys:
                    sample['image_prime'] = target.copy()

            else:
                # only retrieved a single image
                scene, idx = self.items[index]
                image = self._read_single_view(scene, idx)

                sample = {
                    "image": image,
                }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
