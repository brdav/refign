import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from helpers.matching_utils import get_gt_correspondence_mask, warp
from torch import Tensor


class PixelWeightedCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor, pixel_weight: Optional[Tensor] = None) -> Tensor:
        loss = nn.functional.cross_entropy(
            input, target, ignore_index=self.ignore_index, reduction='none')
        if pixel_weight is not None:
            assert pixel_weight.dim() == loss.dim()
            loss = loss * pixel_weight.to(loss.dtype)
        return torch.mean(loss)


class HuberLoss(nn.Module):

    def __init__(self, reduction='mean', delta=1.0):
        super().__init__()
        self.reduction = reduction
        self.delta = delta

    def forward(self, input, target):
        # factor 2 so it makes sense in probabilistic setup
        return 2.0 * nn.functional.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.delta) * self.delta


class MultiScaleFlowLoss(nn.Module):
    """ Module for multi-scale matching loss computation.
    The loss is computed at all estimated flow resolutions and weighted according to level_weights. """

    def __init__(self, level_weights=None, loss_type='L1Loss', downsample_gt_flow=True, reduction='mean'):
        """
        Args:
            level_weights: weights to apply to computed loss at each level (from coarsest to finest pyramid level)
            loss_function: actual loss computation module, used for all levels
            downsample_gt_flow: bool, downsample gt flow to estimated flow resolution? otherwise, the estimated flow
                                of each level is instead up-sampled to ground-truth resolution for loss computation
        """
        super().__init__()
        self.level_weights = level_weights
        self.downsample_gt_flow = downsample_gt_flow
        self.reduction = reduction
        self.loss_type = loss_type
        if loss_type == 'L1Loss':
            self.loss_function = nn.L1Loss(reduction='none')
        elif loss_type == 'L2Loss':
            self.loss_function = nn.MSELoss(reduction='none')
        elif loss_type == 'HuberLoss':
            self.loss_function = HuberLoss(reduction='none')
        else:
            raise ValueError

    def probabilistic_one_scale(self, est_flow, est_uncert, gt_flow, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            mask: valid mask, where the loss is computed. shape (b, H, W)
        """

        if self.downsample_gt_flow:
            h, w = est_flow.shape[-2:]
            gt_flow = nn.functional.interpolate(
                gt_flow, (h, w), mode='bilinear', align_corners=False)
        else:
            h, w = gt_flow.shape[-2:]
            # upsample output to ground truth flow load_size
            est_flow = nn.functional.interpolate(
                est_flow, (h, w), mode='bilinear', align_corners=False)
            est_uncert = nn.functional.interpolate(
                est_uncert, (h, w), mode='bilinear', align_corners=False)

        if mask is not None:
            mask = mask.unsqueeze(1)
            if mask.shape[2] != h or mask.shape[3] != w:
                mask = nn.functional.interpolate(
                    mask.float(), (h, w), mode='bilinear', align_corners=False).floor().bool()
            if not torch.any(mask):
                return est_flow.new_zeros([])

        loss = torch.sum(self.loss_function(est_flow, gt_flow),
                         1, keepdim=True)  # b x 1 x h x w

        # probabilistic part
        assert self.loss_type in ['L2Loss', 'HuberLoss']
        if est_uncert.shape[1] == 1:
            # single warp
            log_var = est_uncert
            loss = 0.5 * torch.exp(-log_var) * loss + \
                log_var + math.log(2 * math.pi)
        elif est_uncert.shape[1] == 2:
            # double warp
            log_var = torch.logsumexp(est_uncert, 1, keepdim=True)
            loss = 0.5 * torch.exp(-log_var) * loss + \
                log_var + math.log(2 * math.pi)
        else:
            raise ValueError

        if self.reduction == 'mean':
            loss = torch.masked_select(loss, mask).mean()
        else:
            raise ValueError
        return loss

    def one_scale(self, est_flow, gt_flow, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            mask: valid mask, where the loss is computed. shape (b, H, W)
        """
        if self.downsample_gt_flow:
            h, w = est_flow.shape[-2:]
            gt_flow = nn.functional.interpolate(
                gt_flow, (h, w), mode='bilinear', align_corners=False)
        else:
            h, w = gt_flow.shape[-2:]
            # upsample output to ground truth flow load_size
            est_flow = nn.functional.interpolate(
                est_flow, (h, w), mode='bilinear', align_corners=False)

        if mask is not None:
            mask = mask.unsqueeze(1)
            if mask.shape[2] != h or mask.shape[3] != w:
                mask = nn.functional.interpolate(
                    mask.float(), (h, w), mode='bilinear', align_corners=False).floor().bool()
            if not torch.any(mask):
                return est_flow.new_zeros([])

        loss = torch.sum(self.loss_function(est_flow, gt_flow),
                         1, keepdim=True)  # b x 1 x h x w
        if self.reduction == 'mean':
            loss = torch.masked_select(loss, mask).mean()
        else:
            raise ValueError
        return loss

    def forward(self, flow_output, gt_flow, mask=None):
        """
        Args:
            network_output: network predictions, can either be a dictionary, where network_output['flow_estimates']
                            is a list containing the estimated flow fields at all levels. or directly the list 
                            of flow fields.
            gt_flow: ground-truth flow field
            mask: bool tensor, valid mask, 1 indicates valid pixels where the loss is computed.
        Returns:
            loss: computed loss
            stats: dict with stats from the loss computation
        """
        if not isinstance(flow_output, Sequence):
            flow_output = [flow_output]

        if self.level_weights:
            level_weights = self.level_weights
        else:
            level_weights = [1] * len(flow_output)

        assert(len(level_weights) == len(flow_output))

        loss = 0
        for i, (flow, weight) in enumerate(zip(flow_output, level_weights)):

            # from smallest load_size to biggest load_size (last one is a quarter of input image load_size
            if mask is not None and isinstance(mask, Sequence):
                mask_used = mask[i]
            else:
                mask_used = mask

            if isinstance(flow, tuple):
                flow, uncert = flow
                level_loss = weight * \
                    self.probabilistic_one_scale(
                        flow, uncert, gt_flow, mask=mask_used)
            else:
                level_loss = weight * \
                    self.one_scale(flow, gt_flow, mask=mask_used)
            loss += level_loss
        return loss


class WBipathLoss(nn.Module):
    """
    Main module computing the W-bipath loss. The W-bipath constraints computes the flow composition from the
    target prime to the target image.
    """

    def __init__(self, objective='multi_scale_flow_loss', reduction='mean', level_weights=None,
                 loss_type='L1Loss', downsample_gt_flow=True, detach_flow_for_warping=True,
                 visibility_mask=False, alpha_1=0.03, alpha_2=0.5):
        """
        Args:
            objective: final objective, like multi-scale EPE or L1 loss
            loss_weight: weights used
            detach_flow_for_warping: bool, prevent back-propagation through the flow used for warping.
            compute_cyclic_consistency:
            alpha_1: hyper-parameter for the visibility mask
            alpha_2: hyper-parameter for the visibility mask
        """
        super().__init__()
        if objective == 'multi_scale_flow_loss':
            self.objective = MultiScaleFlowLoss(level_weights=level_weights,
                                                loss_type=loss_type,
                                                downsample_gt_flow=downsample_gt_flow,
                                                reduction=reduction)
        else:
            raise ValueError
        self.detach_flow_for_warping = detach_flow_for_warping
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.visibility_mask = visibility_mask

    @torch.no_grad()
    def get_cyclic_consistency_mask(self, estimated_flow_target_prime_to_source_per_level,
                                    warping_flow_source_to_target, synthetic_flow):
        h_, w_ = estimated_flow_target_prime_to_source_per_level.shape[-2:]

        synthetic_flow = nn.functional.interpolate(
            synthetic_flow, (h_, w_), mode='bilinear', align_corners=False)

        # defines occluded pixels (or just flow is not good enough)
        mag_sq_fw = self.length_sq(estimated_flow_target_prime_to_source_per_level) + \
            self.length_sq(warping_flow_source_to_target) + \
            self.length_sq(synthetic_flow)
        occ_thresh_fw = self.alpha_1 * mag_sq_fw + self.alpha_2
        fb_occ_fw = self.length_sq(estimated_flow_target_prime_to_source_per_level + warping_flow_source_to_target -
                                   synthetic_flow) > occ_thresh_fw

        # defines the mask of not occluded pixels
        mask_fw = ~fb_occ_fw  # shape bxhxw
        return mask_fw

    @staticmethod
    def length_sq(x):
        return torch.sum(x**2, dim=1)

    def forward(self, estimated_flow_target_prime_to_source,
                estimated_flow_source_to_target, flow_map, mask_used, return_masks=False):
        """
        Args:
            flow_map: corresponds to known flow relating the target prime image to the target
            mask_used: mask indicating in which regions the flow_map is valid
            estimated_flow_target_prime_to_source: list of estimated flows
            estimated_flow_source_to_target: list of estimated flows
        Returns:
            loss_un: final loss
            stats_un: stats from loss computation
            output: dictionary containing some intermediate results, for example the composition flow.
        """
        h, w = flow_map.shape[-2:]  # load_size of the gt flow, meaning load_size of the input images

        estimated_flow_target_prime_to_target_through_composition = []
        masks = []
        mask_cyclic_list = []

        if not isinstance(estimated_flow_target_prime_to_source, Sequence):
            estimated_flow_target_prime_to_source = [
                estimated_flow_target_prime_to_source]
        if not isinstance(estimated_flow_source_to_target, Sequence):
            estimated_flow_source_to_target = [estimated_flow_source_to_target]

        for estimated_flow_target_prime_to_source_per_level, estimated_flow_source_to_target_per_level \
                in zip(estimated_flow_target_prime_to_source, estimated_flow_source_to_target):

            if isinstance(estimated_flow_target_prime_to_source_per_level, tuple):
                probabilistic = True
                estimated_flow_target_prime_to_source_per_level, estimated_uncert_target_prime_to_source_per_level = estimated_flow_target_prime_to_source_per_level
                estimated_flow_source_to_target_per_level, estimated_uncert_source_to_target_per_level = estimated_flow_source_to_target_per_level
            else:
                probabilistic = False

            h_, w_ = estimated_flow_target_prime_to_source_per_level.shape[-2:]

            if self.detach_flow_for_warping:
                estimated_flow_target_prime_to_source_per_level_warping = \
                    estimated_flow_target_prime_to_source_per_level.detach().clone()
            else:
                estimated_flow_target_prime_to_source_per_level_warping = \
                    estimated_flow_target_prime_to_source_per_level.clone()
            estimated_flow_target_prime_to_source_per_level_warping[:, 0, :, :] *= float(
                w_) / float(w)
            estimated_flow_target_prime_to_source_per_level_warping[:, 1, :, :] *= float(
                h_) / float(h)

            warping_flow_source_to_target = warp(estimated_flow_source_to_target_per_level,
                                                 estimated_flow_target_prime_to_source_per_level_warping)
            estimated_flow = estimated_flow_target_prime_to_source_per_level + \
                warping_flow_source_to_target
            if probabilistic:
                # also warp the uncertainty components
                warping_uncert_source_to_target = warp(estimated_uncert_source_to_target_per_level,
                                                       estimated_flow_target_prime_to_source_per_level_warping)
                estimated_uncert = torch.cat(
                    (estimated_uncert_target_prime_to_source_per_level, warping_uncert_source_to_target), 1)
                estimated_flow = (estimated_flow, estimated_uncert)
            estimated_flow_target_prime_to_target_through_composition.append(
                estimated_flow)

            # if estimated flow is inaccurate, this might be all zeros
            mask = get_gt_correspondence_mask(
                estimated_flow_target_prime_to_source_per_level_warping.detach())
            mask = mask & nn.functional.interpolate(mask_used.unsqueeze(1).float(
            ), (h_, w_), mode='bilinear', align_corners=False).squeeze(1).floor().bool() if mask_used is not None else mask

            if self.visibility_mask:
                mask_cyclic = self.get_cyclic_consistency_mask(estimated_flow_target_prime_to_source_per_level.detach(),
                                                               warping_flow_source_to_target.detach(), flow_map)
                mask = mask & mask_cyclic
                mask_cyclic_list.append(mask_cyclic)
            masks.append(mask)

        loss = self.objective(estimated_flow_target_prime_to_target_through_composition,
                              flow_map, mask=masks)

        if return_masks:
            if len(mask_cyclic_list) == 0:
                mask_cyclic_list = None
            return loss, masks, mask_cyclic_list, estimated_flow_target_prime_to_target_through_composition
        return loss
