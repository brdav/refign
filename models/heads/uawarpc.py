import math
import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
from helpers.matching_utils import (unnormalise_and_convert_mapping_to_flow,
                                    warp)

from ..modules import (GlobalFeatureCorrelationLayer,
                       LocalFeatureCorrelationLayer,
                       OpticalFlowEstimatorResidualConnection,
                       RefinementModule, UncertaintyModule)
from .base import BaseHead


class UAWarpCHead(BaseHead):

    def __init__(self,
                 in_index: Union[List[int], int],
                 input_transform: Optional[str] = None,
                 pretrained: Optional[str] = None,
                 batch_norm: bool = True,
                 refinement_at_adaptive_res: bool = True,
                 refinement_at_finest_level: bool = True,
                 estimate_uncertainty: bool = False,
                 uncertainty_mixture: bool = False,
                 iterative_refinement: bool = False):
        super().__init__(None, in_index, input_transform)
        self.estimate_uncertainty = estimate_uncertainty
        self.uncertainty_mixture = uncertainty_mixture
        self.iterative_refinement = iterative_refinement
        self.global_corr = GlobalFeatureCorrelationLayer(
            cyclic_consistency=True)
        self.local_corr = LocalFeatureCorrelationLayer(
            patch_size=9)

        # level 4, 16x16, global correlation
        nd = 16 * 16
        od = nd
        self.decoder4 = OpticalFlowEstimatorResidualConnection(
            in_channels=od, batch_norm=batch_norm, output_x=True)

        # level 3, 32x32, constrained correlation, patchsize 9
        nd = 9 * 9
        od = nd + 2
        if self.estimate_uncertainty:
            od += 1  # input also the upsampled log_var of previous resolution
        self.decoder3 = OpticalFlowEstimatorResidualConnection(
            in_channels=od, batch_norm=batch_norm, output_x=True)

        self.refinement_at_adaptive_res = refinement_at_adaptive_res
        if self.refinement_at_adaptive_res:
            self.refinement_module_adaptive = RefinementModule(
                32, batch_norm=batch_norm)

        nbr_upfeat_channels = 2

        od = nd + 2
        if self.estimate_uncertainty:
            od += 1  # input also the upsampled log_var of previous resolution
        self.decoder2 = OpticalFlowEstimatorResidualConnection(
            in_channels=od, batch_norm=batch_norm, output_x=True)

        self.reduce = nn.Conv2d(32, nbr_upfeat_channels,
                                kernel_size=1, bias=True)

        od = nd + 2 + nbr_upfeat_channels
        if self.estimate_uncertainty:
            od += 1  # input also the upsampled log_var of previous resolution
        self.decoder1 = OpticalFlowEstimatorResidualConnection(
            in_channels=od, batch_norm=batch_norm, output_x=True)

        self.refinement_at_finest_level = refinement_at_finest_level
        if self.refinement_at_finest_level:
            self.refinement_module_finest = RefinementModule(
                32, batch_norm=batch_norm)

        if self.estimate_uncertainty:
            self.estimate_uncertainty_components4 = UncertaintyModule(in_channels=1,
                                                                      search_size=16)
            self.estimate_uncertainty_components3 = UncertaintyModule(in_channels=1,
                                                                      search_size=9,
                                                                      feed_in_previous=True)
            self.estimate_uncertainty_components2 = UncertaintyModule(in_channels=1,
                                                                      search_size=9,
                                                                      feed_in_previous=True)
            self.estimate_uncertainty_components1 = UncertaintyModule(in_channels=1,
                                                                      search_size=9,
                                                                      feed_in_previous=True)

        if pretrained is not None:
            self.load_weights(pretrained)

    def forward(self, trg, src, trg_256, src_256, out_size):
        c11, c12 = self._transform_inputs(trg)
        c13, c14 = self._transform_inputs(trg_256)
        c21, c22 = self._transform_inputs(src)
        c23, c24 = self._transform_inputs(src_256)

        c11 = nn.functional.normalize(c11, p=2, dim=1)
        c12 = nn.functional.normalize(c12, p=2, dim=1)
        c13 = nn.functional.normalize(c13, p=2, dim=1)
        c14 = nn.functional.normalize(c14, p=2, dim=1)
        c21 = nn.functional.normalize(c21, p=2, dim=1)
        c22 = nn.functional.normalize(c22, p=2, dim=1)
        c23 = nn.functional.normalize(c23, p=2, dim=1)
        c24 = nn.functional.normalize(c24, p=2, dim=1)

        h_256, w_256 = (256, 256)

        # level 4: 16x16
        h_4, w_4 = c14.shape[-2:]
        assert (h_4, w_4) == (16, 16), (h_4, w_4)
        corr4 = self.global_corr(c24, c14)

        # init_map = corr4.new_zeros(size=(b, 2, h_4, w_4))
        est_map4, x4 = self.decoder4(corr4)
        flow4_256 = unnormalise_and_convert_mapping_to_flow(est_map4)
        # scale flow values to 256x256
        flow4_256[:, 0, :, :] *= w_256 / float(w_4)
        flow4_256[:, 1, :, :] *= h_256 / float(h_4)

        if self.estimate_uncertainty:
            uncert_components4_256 = self.estimate_uncertainty_components4(
                corr4, x4)
            # scale uncert values to 256
            assert w_256 / float(w_4) == h_256 / float(h_4)
            uncert_components4_256[:, 0, :, :] += 2 * \
                math.log(w_256 / float(w_4))

        # level 3: 32x32
        h_3, w_3 = c13.shape[-2:]
        assert (h_3, w_3) == (32, 32), (h_3, w_3)
        up_flow4 = nn.functional.interpolate(input=flow4_256, size=(
            h_3, w_3), mode='bilinear', align_corners=False)
        if self.estimate_uncertainty:
            up_uncert_components4 = nn.functional.interpolate(
                input=uncert_components4_256, size=(h_3, w_3), mode='bilinear', align_corners=False)

        # for warping, we need flow values at 32x32 scale
        up_flow_4_warping = up_flow4.clone()
        up_flow_4_warping[:, 0, :, :] *= w_3 / float(w_256)
        up_flow_4_warping[:, 1, :, :] *= h_3 / float(h_256)
        warp3 = warp(c23, up_flow_4_warping)

        # constrained correlation
        corr3 = self.local_corr(warp3, c13)
        if self.estimate_uncertainty:
            inp_flow_dec3 = torch.cat(
                (corr3, up_flow4, up_uncert_components4), 1)
        else:
            inp_flow_dec3 = torch.cat((corr3, up_flow4), 1)
        res_flow3, x3 = self.decoder3(inp_flow_dec3)
        if self.refinement_at_adaptive_res:
            res_flow3 = res_flow3 + self.refinement_module_adaptive(x3)
        flow3 = res_flow3 + up_flow4

        if self.estimate_uncertainty:
            uncert_components3 = self.estimate_uncertainty_components3(
                corr3, x3, up_uncert_components4, up_flow4)

        # change from absolute resolutions to relative resolutions
        # scale flow4 and flow3 magnitude to original resolution
        h_original, w_original = out_size
        flow3[:, 0, :, :] *= w_original / float(w_256)
        flow3[:, 1, :, :] *= h_original / float(h_256)
        if self.estimate_uncertainty:
            # APPROXIMATION FOR NON-SQUARE IMAGES --> use the diagonal
            diag_original = math.sqrt(h_original ** 2 + w_original ** 2)
            diag_256 = math.sqrt(h_256 ** 2 + w_256 ** 2)
            uncert_components3[:, 0, :, :] += 2 * \
                math.log(diag_original / float(diag_256))

        if self.iterative_refinement and not self.training:
            # from 32x32 resolution, if upsampling to 1/8*original resolution is too big,
            # do iterative upsampling so that gap is always smaller than 2.
            R = float(max(h_original, w_original)) / 8.0 / 32.0
            minimum_ratio = 3.0  # if the image is >= 1086 in one dimension, do refinement
            nbr_extra_layers = max(
                0, int(round(math.log(R / minimum_ratio) / math.log(2))))
            if nbr_extra_layers > 0:
                for n in range(nbr_extra_layers):
                    ratio = 1.0 / (8.0 * 2 ** (nbr_extra_layers - n))
                    h_this = int(h_original * ratio)
                    w_this = int(w_original * ratio)
                    up_flow3 = nn.functional.interpolate(input=flow3, size=(
                        h_this, w_this), mode='bilinear', align_corners=False)
                    if self.estimate_uncertainty:
                        up_uncert_components3 = nn.functional.interpolate(input=uncert_components3, size=(
                            h_this, w_this), mode='bilinear', align_corners=False)
                    c23_bis = nn.functional.interpolate(
                        c22, size=(h_this, w_this), mode='area')
                    c13_bis = nn.functional.interpolate(
                        c12, size=(h_this, w_this), mode='area')
                    warp3 = warp(c23_bis, up_flow3 * ratio)
                    corr3 = self.local_corr(warp3, c13_bis)
                    if self.estimate_uncertainty:
                        inp_flow_dec3 = torch.cat(
                            (corr3, up_flow3, up_uncert_components3), 1)
                    else:
                        inp_flow_dec3 = torch.cat((corr3, up_flow3), 1)
                    res_flow3, x3 = self.decoder2(inp_flow_dec3)
                    flow3 = res_flow3 + up_flow3
                    if self.estimate_uncertainty:
                        uncert_components3 = self.estimate_uncertainty_components2(
                            corr3, x3, up_uncert_components3, up_flow3)

        # level 2: 1/8 of original resolution
        h_2, w_2 = c12.shape[-2:]
        up_flow3 = nn.functional.interpolate(input=flow3, size=(
            h_2, w_2), mode='bilinear', align_corners=False)
        if self.estimate_uncertainty:
            up_uncert_components3 = nn.functional.interpolate(
                input=uncert_components3, size=(h_2, w_2), mode='bilinear', align_corners=False)

        up_flow_3_warping = up_flow3.clone()
        up_flow_3_warping[:, 0, :, :] *= w_2 / float(w_original)
        up_flow_3_warping[:, 1, :, :] *= h_2 / float(h_original)
        warp2 = warp(c22, up_flow_3_warping)

        # constrained correlation
        corr2 = self.local_corr(warp2, c12)
        if self.estimate_uncertainty:
            inp_flow_dec2 = torch.cat(
                (corr2, up_flow3, up_uncert_components3), 1)
        else:
            inp_flow_dec2 = torch.cat((corr2, up_flow3), 1)
        res_flow2, x2 = self.decoder2(inp_flow_dec2)
        flow2 = res_flow2 + up_flow3

        if self.estimate_uncertainty:
            uncert_components2 = self.estimate_uncertainty_components2(
                corr2, x2, up_uncert_components3, up_flow3)

        # level 1: 1/4 of original resolution
        h_1, w_1 = c11.shape[-2:]
        up_flow2 = nn.functional.interpolate(input=flow2, size=(
            h_1, w_1), mode='bilinear', align_corners=False)
        if self.estimate_uncertainty:
            up_uncert_components2 = nn.functional.interpolate(
                input=uncert_components2, size=(h_1, w_1), mode='bilinear', align_corners=False)

        up_feat2 = nn.functional.interpolate(input=x2, size=(
            h_1, w_1), mode='bilinear', align_corners=False)
        up_feat2 = self.reduce(up_feat2)

        up_flow_2_warping = up_flow2.clone()
        up_flow_2_warping[:, 0, :, :] *= w_1 / float(w_original)
        up_flow_2_warping[:, 1, :, :] *= h_1 / float(h_original)
        warp1 = warp(c21, up_flow_2_warping)

        corr1 = self.local_corr(warp1, c11)
        if self.estimate_uncertainty:
            inp_flow_dec1 = torch.cat(
                (corr1, up_flow2, up_feat2, up_uncert_components2), 1)
        else:
            inp_flow_dec1 = torch.cat((corr1, up_flow2, up_feat2), 1)
        res_flow1, x = self.decoder1(inp_flow_dec1)
        if self.refinement_at_finest_level:
            res_flow1 = res_flow1 + self.refinement_module_finest(x)
        flow1 = res_flow1 + up_flow2

        # upscale also flow4
        flow4 = flow4_256.clone()
        flow4[:, 0, :, :] *= w_original / float(w_256)
        flow4[:, 1, :, :] *= h_original / float(h_256)

        if self.estimate_uncertainty:
            uncert_components1 = self.estimate_uncertainty_components1(
                corr1, x, up_uncert_components2, up_flow2)

            # APPROXIMATION FOR NON-SQUARE IMAGES --> use the diagonal
            uncert_components4 = uncert_components4_256.clone()
            uncert_components4[:, 0, :, :] += 2 * \
                math.log(diag_original / float(diag_256))

            return (flow4, uncert_components4), (flow3, uncert_components3), (flow2, uncert_components2), (flow1, uncert_components1)

        return flow4, flow3, flow2, flow1

    def load_weights(self, pretrain_path):
        if pretrain_path is None:
            return
        if os.path.exists(pretrain_path):
            checkpoint = torch.load(
                pretrain_path, map_location=lambda storage, loc: storage)
        elif os.path.exists(os.path.join(os.environ.get('TORCH_HOME', ''), 'hub', pretrain_path)):
            checkpoint = torch.load(os.path.join(os.environ.get(
                'TORCH_HOME', ''), 'hub', pretrain_path), map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                pretrain_path, progress=True, map_location=lambda storage, loc: storage)
        if 'state_dict' in checkpoint.keys():
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('alignment_head.'):
                new_k = k.replace('alignment_head.', '')
            else:
                continue  # ignore the rest
            new_state_dict[new_k] = v
        self.load_state_dict(new_state_dict, strict=True)
