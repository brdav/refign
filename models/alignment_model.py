import os
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from helpers.matching_utils import \
    estimate_probability_of_confidence_interval_of_mixture_density
from helpers.metrics import MyMetricCollection
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, instantiate_class

from .heads.base import BaseHead


@MODEL_REGISTRY
class AlignmentModel(pl.LightningModule):
    def __init__(self,
                 optimizer_init: dict,
                 lr_scheduler_init: dict,
                 alignment_backbone: nn.Module,
                 alignment_head: BaseHead,
                 selfsupervised_loss: nn.Module,
                 unsupervised_loss: nn.Module,
                 metrics: dict = {},
                 apply_constant_flow_weights: bool = False,
                 pretrained: Optional[str] = None,
                 ):
        super().__init__()
        #### MODEL ####
        self.alignment_backbone = alignment_backbone
        self.alignment_head = alignment_head
        for param in self.alignment_backbone.parameters():
            param.requires_grad = False

        #### LOSSES ####
        self.selfsupervised_loss = selfsupervised_loss
        self.unsupervised_loss = unsupervised_loss
        self.apply_constant_flow_weights = apply_constant_flow_weights

        #### METRICS ####
        val_metrics = {'val_{}_{}'.format(ds, el['class_path'].split(
            '.')[-1]): instantiate_class(tuple(), el) for ds, m in metrics.get('val', {}).items() for el in m}
        test_metrics = {'test_{}_{}'.format(ds, el['class_path'].split(
            '.')[-1]): instantiate_class(tuple(), el) for ds, m in metrics.get('test', {}).items() for el in m}
        self.valid_metrics = MyMetricCollection(val_metrics)
        self.test_metrics = MyMetricCollection(test_metrics)

        #### OPTIMIZATION ####
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init

        #### LOAD WEIGHTS ####
        self.load_weights(pretrained)

    def forward(self, images_i, images_j):
        b, _, h, w = images_i.shape
        images_i_256 = nn.functional.interpolate(
            images_i, size=(256, 256), mode='area')
        images_j_256 = nn.functional.interpolate(
            images_j, size=(256, 256), mode='area')
        with torch.no_grad():
            x_backbone = self.alignment_backbone(
                torch.cat([images_j, images_i]), extract_only_indices=[-3, -2])
            x_backbone_256 = self.alignment_backbone(
                torch.cat([images_j_256, images_i_256]), extract_only_indices=[-2, -1])
            unpacked_x = [torch.split(l, [b, b]) for l in x_backbone]
            pyr_j, pyr_i = zip(*unpacked_x)
            unpacked_x_256 = [torch.split(l, [b, b]) for l in x_backbone_256]
            pyr_j_256, pyr_i_256 = zip(*unpacked_x_256)
        flow_i_to_j, uncert_i_to_j = self.alignment_head(
            pyr_i, pyr_j, pyr_i_256, pyr_j_256, (h, w))[-1]
        uncert_i_to_j = nn.functional.interpolate(
            uncert_i_to_j, size=(h, w), mode='bilinear', align_corners=False)
        flow_i_to_j = nn.functional.interpolate(
            flow_i_to_j, size=(h, w), mode='bilinear', align_corners=False)
        uncert_i_to_j = 1.0 - \
            estimate_probability_of_confidence_interval_of_mixture_density(
                uncert_i_to_j, R=1.0)
        return flow_i_to_j, uncert_i_to_j

    def training_step(self, batch, batch_idx):
        images_ref = batch['image_ref']
        images_trg = batch['image_trg']
        images_prime = batch['image_prime']
        images_prime_flow = batch['flow_prime']
        images_prime_mask = batch['mask_prime']

        b, _, h, w = images_trg.shape
        images_trg_256 = nn.functional.interpolate(
            images_trg, size=(256, 256), mode='area')
        images_ref_256 = nn.functional.interpolate(
            images_ref, size=(256, 256), mode='area')
        images_prime_256 = nn.functional.interpolate(
            images_prime, size=(256, 256), mode='area')

        with torch.no_grad():
            x_backbone = self.alignment_backbone(torch.cat(
                [images_ref, images_trg, images_prime]), extract_only_indices=[-3, -2])
            unpacked_x = [torch.split(l, [b, b, b]) for l in x_backbone]
            pyr_ref, pyr_trg, pyr_prime = zip(*unpacked_x)
            x_backbone_256 = self.alignment_backbone(torch.cat(
                [images_ref_256, images_trg_256, images_prime_256]), extract_only_indices=[-2, -1])
            unpacked_x_256 = [torch.split(l, [b, b, b])
                              for l in x_backbone_256]
            pyr_ref_256, pyr_trg_256, pyr_prime_256 = zip(*unpacked_x_256)

            pyr_l = (pyr_ref, pyr_trg)
            pyr_l_256 = (pyr_ref_256, pyr_trg_256)
            pyr_i = []
            pyr_j = []
            pyr_i_256 = []
            pyr_j_256 = []
            for l in range(len(pyr_ref)):
                pyr_i.append(torch.stack(
                    [pyr_l[b][l][idx] for idx, b in enumerate(batch['prime_trg_idx'])]))
                pyr_j.append(torch.stack(
                    [pyr_l[1 - b][l][idx] for idx, b in enumerate(batch['prime_trg_idx'])]))
                pyr_i_256.append(torch.stack(
                    [pyr_l_256[b][l][idx] for idx, b in enumerate(batch['prime_trg_idx'])]))
                pyr_j_256.append(torch.stack(
                    [pyr_l_256[1 - b][l][idx] for idx, b in enumerate(batch['prime_trg_idx'])]))

        # warp supervision
        prime_i_flow = self.alignment_head(
            pyr_prime, pyr_i, pyr_prime_256, pyr_i_256, (h, w))

        # bipath
        prime_j_flow = self.alignment_head(
            pyr_prime, pyr_j, pyr_prime_256, pyr_j_256, (h, w))
        j_i_flow = self.alignment_head(
            pyr_j, pyr_i, pyr_j_256, pyr_i_256, (h, w))

        ss_loss = self.selfsupervised_loss(
            prime_i_flow, images_prime_flow, mask=images_prime_mask)
        us_loss = self.unsupervised_loss(prime_j_flow,
                                         j_i_flow,
                                         images_prime_flow,
                                         mask_used=images_prime_mask)

        # slight difference to OG GLUNet, where loss weighting is done separately for 256 scale
        weight_ss, weight_us = self.weights_selfsupervised_and_unsupervised(ss_loss,
                                                                            us_loss,
                                                                            self.apply_constant_flow_weights)
        loss = weight_ss * ss_loss + weight_us * us_loss
        self.log("train_matching_loss", loss, batch_size=b)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images_ref = batch['image_ref']
        images_trg = batch['image']
        corr_pts_ref = batch['corr_pts_ref']
        corr_pts_trg = batch['corr_pts']

        h, w = images_ref.shape[-2:]
        flow_trg_to_ref, uncert_trg_to_ref = self.forward(
            images_trg, images_ref)

        src_name = self.trainer.datamodule.idx_to_name['val'][dataloader_idx]
        for k, m in self.valid_metrics.items():
            if src_name in k:
                m(flow_trg_to_ref, corr_pts_ref,
                  corr_pts_trg, (h, w), uncert_trg_to_ref)

    def validation_epoch_end(self, outs):
        out_dict = self.valid_metrics.compute()
        self.valid_metrics.reset()
        for k, v in out_dict.items():
            self.log(k, v)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images_ref = batch['image_ref']
        images_trg = batch['image']
        corr_pts_ref = batch['corr_pts_ref']
        corr_pts_trg = batch['corr_pts']

        h, w = images_ref.shape[-2:]
        flow_trg_to_ref, uncert_trg_to_ref = self.forward(
            images_trg, images_ref)

        src_name = self.trainer.datamodule.idx_to_name['test'][dataloader_idx]
        for k, m in self.test_metrics.items():
            if src_name in k:
                m(flow_trg_to_ref, corr_pts_ref,
                  corr_pts_trg, (h, w), uncert_trg_to_ref)

    def test_epoch_end(self, outs):
        out_dict = self.test_metrics.compute()
        self.test_metrics.reset()
        for k, v in out_dict.items():
            self.log(k, v)

    def configure_optimizers(self):
        optimizer = instantiate_class(
            filter(lambda p: p.requires_grad, self.parameters()), self.optimizer_init)
        lr_scheduler = instantiate_class(optimizer, self.lr_scheduler_init)
        lr_scheduler_config = {'scheduler': lr_scheduler,
                               'interval': 'step'}
        return [optimizer], [lr_scheduler_config]

    def load_weights(self, pretrain_path):
        if pretrain_path is None:
            return
        if os.path.exists(pretrain_path):
            checkpoint = torch.load(pretrain_path, map_location=self.device)
        elif os.path.exists(os.path.join(os.environ.get('TORCH_HOME', ''), 'hub', pretrain_path)):
            checkpoint = torch.load(os.path.join(os.environ.get(
                'TORCH_HOME', ''), 'hub', pretrain_path), map_location=self.device)
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                pretrain_path, progress=True, map_location=self.device)
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=True)

    @staticmethod
    @torch.no_grad()
    def weights_selfsupervised_and_unsupervised(loss_ss, loss_un, weight_ss=1.0, weight_un=1.0, apply_constant_weights=False):
        if not apply_constant_weights:
            ratio = weight_ss / weight_un
            if loss_un > loss_ss:
                u_l_w = 1.0
                s_l_w = torch.clamp(
                    loss_un / loss_ss.clamp(min=1e-8) * ratio, max=100).item()
            else:
                u_l_w = torch.clamp(
                    loss_ss / loss_un.clamp(min=1e-8) / ratio, max=100).item()
                s_l_w = 1.0
            return s_l_w, u_l_w
        else:
            return weight_ss, weight_un

    def train(self, mode=True):
        super().train(mode=mode)
        for m in self.alignment_backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
