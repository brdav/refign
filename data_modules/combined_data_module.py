import os
import random
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.cli import (DATAMODULE_REGISTRY,
                                             instantiate_class)
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from . import transforms as transform_lib
from .datasets import *

DATA_DIR = os.environ['DATA_DIR']


def my_collate(batch):
    elem = batch[0]
    assert isinstance(elem, dict)
    ignore_keys = ['corr_pts_ref', 'corr_pts']
    batch_out = {}
    for key in elem:
        if key in ignore_keys:
            batch_out[key] = [torch.as_tensor(d[key]) for d in batch]
        else:
            batch_out[key] = default_collate([d[key] for d in batch])
    return batch_out


@DATAMODULE_REGISTRY
class CombinedDataModule(LightningDataModule):

    def __init__(
        self,
        load_config: dict,
        num_workers: int = 0,
        batch_size: int = 8,
        batch_size_divisor: int = 1,
        pin_memory: bool = True,
        debug: bool = False,
        ignore_every_second_semantic_training_batch: bool = False,
    ) -> None:
        super().__init__()
        self.debug = debug
        self.data_dirs = {
            'ACDC': os.path.join(DATA_DIR, 'ACDC'),
            'Cityscapes': os.path.join(DATA_DIR, 'Cityscapes'),
            'DarkZurich': os.path.join(DATA_DIR, 'DarkZurich'),
            'NighttimeDriving': os.path.join(DATA_DIR, 'NighttimeDrivingTest'),
            'BDD100kNight': os.path.join(DATA_DIR, 'bdd100k'),
            'RobotCar': os.path.join(DATA_DIR, 'RobotCar'),
            'MegaDepth': os.path.join(DATA_DIR, 'MegaDepth_debug' if self.debug else 'MegaDepth'),
            'RobotCarMatching': os.path.join(DATA_DIR, 'RobotCar'),
        }
        self.num_workers = num_workers
        assert batch_size % batch_size_divisor == 0
        self.batch_size_divisor = batch_size_divisor
        self.batch_size = batch_size // batch_size_divisor
        self.pin_memory = pin_memory
        self.ignore_every_second_semantic_training_batch = ignore_every_second_semantic_training_batch

        self.train_on = []
        self.train_config = []
        self.val_on = []
        self.val_config = []
        self.test_on = []
        self.test_config = []
        self.predict_on = []
        self.predict_config = []

        # parse load_config
        if 'train' in load_config:
            for ds, conf in load_config['train'].items():
                if isinstance(conf, dict):
                    self.train_on.append(ds)
                    self.train_config.append(conf)
                elif isinstance(conf, list):
                    for el in conf:
                        self.train_on.append(ds)
                        self.train_config.append(el)

        if 'val' in load_config:
            for ds, conf in load_config['val'].items():
                if isinstance(conf, dict):
                    self.val_on.append(ds)
                    self.val_config.append(conf)
                elif isinstance(conf, list):
                    for el in conf:
                        self.val_on.append(ds)
                        self.val_config.append(el)

        if 'test' in load_config:
            for ds, conf in load_config['test'].items():
                if isinstance(conf, dict):
                    self.test_on.append(ds)
                    self.test_config.append(conf)
                elif isinstance(conf, list):
                    for el in conf:
                        self.test_on.append(ds)
                        self.test_config.append(el)

        if 'predict' in load_config:
            for ds, conf in load_config['predict'].items():
                if isinstance(conf, dict):
                    self.predict_on.append(ds)
                    self.predict_config.append(conf)
                elif isinstance(conf, list):
                    for el in conf:
                        self.predict_on.append(ds)
                        self.predict_config.append(el)

        self.idx_to_name = {'train': {}, 'val': {}, 'test': {}, 'predict': {}}
        for idx, ds in enumerate(self.train_on):
            self.idx_to_name['train'][idx] = ds
        for idx, ds in enumerate(self.val_on):
            self.idx_to_name['val'][idx] = ds
        for idx, ds in enumerate(self.test_on):
            self.idx_to_name['test'][idx] = ds
        for idx, ds in enumerate(self.predict_on):
            self.idx_to_name['predict'][idx] = ds

        if len(self.train_on) > 0:
            assert self.batch_size % len(
                self.train_on) == 0, 'batch size should be divisible by number of train datasets'

        # handle transformations
        for idx, (ds, cfg) in enumerate(zip(self.train_on, self.train_config)):
            trafos = cfg.pop('transforms', None)
            if trafos:
                self.train_config[idx]['transforms'] = transform_lib.Compose(
                    [instantiate_class(tuple(), t) for t in trafos])
            else:
                self.train_config[idx]['transforms'] = transform_lib.ToTensor()
        for idx, (ds, cfg) in enumerate(zip(self.val_on, self.val_config)):
            trafos = cfg.pop('transforms', None)
            if trafos:
                self.val_config[idx]['transforms'] = transform_lib.Compose(
                    [instantiate_class(tuple(), t) for t in trafos])
            else:
                self.val_config[idx]['transforms'] = transform_lib.ToTensor()
        for idx, (ds, cfg) in enumerate(zip(self.test_on, self.test_config)):
            trafos = cfg.pop('transforms', None)
            if trafos:
                self.test_config[idx]['transforms'] = transform_lib.Compose(
                    [instantiate_class(tuple(), t) for t in trafos])
            else:
                self.test_config[idx]['transforms'] = transform_lib.ToTensor()
        for idx, (ds, cfg) in enumerate(zip(self.predict_on, self.predict_config)):
            trafos = cfg.pop('transforms', None)
            if trafos:
                self.predict_config[idx]['transforms'] = transform_lib.Compose(
                    [instantiate_class(tuple(), t) for t in trafos])
            else:
                self.predict_config[idx]['transforms'] = transform_lib.ToTensor(
                )

        self.val_batch_size = max(
            1, self.batch_size // max(len(self.train_on), 1) // 2)
        self.test_batch_size = 1

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_ds = []
            for ds, cfg in zip(self.train_on, self.train_config):
                self.train_ds.append(globals()[ds](
                    self.data_dirs[ds],
                    stage="train",
                    **cfg,
                    debug=self.debug
                ))

        if stage in (None, "fit", "validate"):
            self.val_ds = []
            for ds, cfg in zip(self.val_on, self.val_config):
                self.val_ds.append(globals()[ds](
                    self.data_dirs[ds],
                    stage="val",
                    **cfg,
                    debug=self.debug
                ))

        if stage in (None, "test"):
            self.test_ds = []
            for ds, cfg in zip(self.test_on, self.test_config):
                self.test_ds.append(globals()[ds](
                    self.data_dirs[ds],
                    stage="test",
                    **cfg,
                    debug=self.debug
                ))

        if stage in (None, "predict"):
            self.predict_ds = []
            for ds, cfg in zip(self.predict_on, self.predict_config):
                self.predict_ds.append(globals()[ds](
                    self.data_dirs[ds],
                    stage="predict",
                    **cfg,
                    debug=self.debug
                ))

    def train_dataloader(self):
        loader_list = []
        for ds in self.train_ds:
            loader = DataLoader(
                dataset=ds,
                batch_size=self.batch_size // len(self.train_on),
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=my_collate,
                drop_last=True,
                pin_memory=self.pin_memory,
            )
            loader_list.append(loader)
        return loader_list

    def val_dataloader(self):
        loader_list = []
        for ds in self.val_ds:
            loader = DataLoader(
                dataset=ds,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=my_collate,
                pin_memory=self.pin_memory,
                drop_last=False,
            )
            loader_list.append(loader)
        return loader_list

    def test_dataloader(self):
        loader_list = []
        for ds in self.test_ds:
            loader = DataLoader(
                dataset=ds,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=my_collate,
                pin_memory=self.pin_memory,
                drop_last=False,
            )
            loader_list.append(loader)
        return loader_list

    def predict_dataloader(self):
        loader_list = []
        for ds in self.predict_ds:
            loader = DataLoader(
                dataset=ds,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=my_collate,
                pin_memory=self.pin_memory,
                drop_last=False,
            )
            loader_list.append(loader)
        return loader_list

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            new_batch = {}
            src_inp = []
            src_y = []
            trg_inp = []
            ref_inp = []
            prime_inp = []
            prime_flow = []
            prime_mask = []
            prime_trg_idx = []
            for sub_batch in batch:
                if 'semantic' in sub_batch.keys():  # supervised semantic batch
                    src_inp.append(sub_batch['image'])
                    src_y.append(sub_batch['semantic'])
                else:  # domain adaptation or corr batch
                    if 'image' in sub_batch.keys():
                        trg_inp.append(sub_batch['image'])
                    if 'image_ref' in sub_batch.keys():
                        ref_inp.append(sub_batch['image_ref'])
                    if 'image_prime' in sub_batch.keys():
                        prime_inp.append(sub_batch['image_prime'])
                        prime_flow.append(sub_batch['image_prime_flow'])
                        prime_mask.append(sub_batch['image_prime_mask'])
                        prime_trg_idx.append(sub_batch['image_prime_idx'])
            if len(src_inp) > 0:
                new_batch['image_src'] = torch.cat(src_inp, dim=0)
                new_batch['semantic_src'] = torch.cat(src_y, dim=0)
            if len(trg_inp) > 0:
                new_batch['image_trg'] = torch.cat(trg_inp, dim=0)
            if len(ref_inp) > 0:
                new_batch['image_ref'] = torch.cat(ref_inp, dim=0)
            if len(prime_inp) > 0:
                assert len(prime_inp) == len(ref_inp)
                new_batch['image_prime'] = torch.cat(prime_inp, dim=0)
                new_batch['flow_prime'] = torch.cat(prime_flow, dim=0)
                new_batch['mask_prime'] = torch.cat(prime_mask, dim=0)
                new_batch['prime_trg_idx'] = torch.cat(prime_trg_idx, dim=0)
            if self.ignore_every_second_semantic_training_batch:
                assert len(new_batch['image_src']) > self.batch_size // len(
                    self.train_on), 'can only ignore in semi-supervised case'
                if random.random() < 0.5:
                    src_len = len(new_batch['image_src'])
                    new_batch['image_src'] = new_batch['image_src'][:src_len // 2]
                    new_batch['semantic_src'] = new_batch['semantic_src'][:src_len // 2]
            return new_batch
        else:
            return batch
