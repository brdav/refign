import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch import Tensor
from torchmetrics.functional.classification.confusion_matrix import \
    _confusion_matrix_update


class MyMetricCollection(torchmetrics.MetricCollection):

    def compute(self) -> Dict[str, Any]:
        """Compute the result for each metric in the collection."""
        res = {k: m.compute()
               for k, m in self.items(keep_base=True, copy_state=False)}
        res = self._my_flatten_dict(res)
        return {self._set_name(k): v for k, v in res.items()}

    @staticmethod
    def _my_flatten_dict(x: Dict) -> Dict:
        """Flatten dict of dicts into single dict."""
        new_dict = {}
        for key, value in x.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    new_dict[key + '_' + k] = v
            else:
                new_dict[key] = value
        return new_dict


class SparseEPE(torchmetrics.Metric):

    full_state_update: bool = False

    def __init__(self, uncertainty_estimation=False, **kwargs):
        super().__init__(**kwargs)
        self.add_state("AEPE", default=torch.tensor(
            0, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("PCK_1", default=torch.tensor(
            0, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("PCK_3", default=torch.tensor(
            0, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("PCK_5", default=torch.tensor(
            0, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("PCK_10", default=torch.tensor(
            0, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("nbr_valid_corr", default=torch.tensor(
            0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("nbr_samples", default=torch.tensor(
            0, dtype=torch.long), dist_reduce_fx="sum")
        self.uncertainty_estimation = uncertainty_estimation
        if self.uncertainty_estimation:
            self.add_state("AUSE_AEPE", default=torch.tensor(
                0, dtype=torch.double), dist_reduce_fx="sum")

    def update(self, t_s_flow: Tensor, corr_pts_s: List[Tensor], corr_pts_t: List[Tensor], out_size: Sequence, uncertainty_est: Optional[Tensor] = None):
        h, w = out_size

        # resizing should be done before
        assert (t_s_flow.shape[-2], t_s_flow.shape[-1]) == (h, w)

        for bb in range(t_s_flow.shape[0]):
            x_s = corr_pts_s[bb][:, 0]
            y_s = corr_pts_s[bb][:, 1]
            x_t = corr_pts_t[bb][:, 0]
            y_t = corr_pts_t[bb][:, 1]

            # filter correspondences, remove the ones for which xB, yB are not in image
            index_valid_s = (torch.round(x_s) >= 0) * (torch.round(x_s)
                                                       < w) * (torch.round(y_s) >= 0) * (torch.round(y_s) < h)
            index_valid_t = (torch.round(x_t) >= 0) * (torch.round(x_t)
                                                       < w) * (torch.round(y_t) >= 0) * (torch.round(y_t) < h)
            index_valid = index_valid_s * index_valid_t
            x_s, y_s, x_t, y_t = x_s[index_valid], y_s[index_valid], x_t[index_valid], y_t[index_valid]
            nbr_valid_corr = index_valid.sum()

            # calculates the PCK
            if nbr_valid_corr > 0:
                flow_gt_x = x_s - x_t
                flow_gt_y = y_s - y_t
                flow_est_x = t_s_flow[bb, 0, torch.round(
                    y_t).long(), torch.round(x_t).long()]
                flow_est_y = t_s_flow[bb, 1, torch.round(
                    y_t).long(), torch.round(x_t).long()]
                EPE = ((flow_gt_x - flow_est_x) ** 2 +
                       (flow_gt_y - flow_est_y) ** 2) ** 0.5
                AEPE = torch.mean(EPE)
                PCK_1 = torch.sum(EPE <= 1.0)
                PCK_3 = torch.sum(EPE <= 3.0)
                PCK_5 = torch.sum(EPE <= 5.0)
                PCK_10 = torch.sum(EPE <= 10.0)
                self.AEPE += AEPE
                self.PCK_1 += PCK_1
                self.PCK_3 += PCK_3
                self.PCK_5 += PCK_5
                self.PCK_10 += PCK_10
                self.nbr_valid_corr += nbr_valid_corr
                self.nbr_samples += 1

                if self.uncertainty_estimation:
                    flow_est = torch.stack([flow_est_x, flow_est_y], dim=1)
                    flow_gt = torch.stack([flow_gt_x, flow_gt_y], dim=1)
                    uncert = uncertainty_est[bb, 0, torch.round(
                        y_t).long(), torch.round(x_t).long()]
                    uncert_dict = self.compute_aucs(flow_gt, flow_est, uncert)
                    AUSE_AEPE = uncert_dict['EPE']
                    self.AUSE_AEPE += AUSE_AEPE

    def compute(self):
        out_dict = {
            'AEPE': self.AEPE / self.nbr_samples.double(),
            'PCK_1': self.PCK_1 / self.nbr_valid_corr.double(),
            'PCK_3': self.PCK_3 / self.nbr_valid_corr.float(),
            'PCK_5': self.PCK_5 / self.nbr_valid_corr.double(),
            'PCK_10': self.PCK_10 / self.nbr_valid_corr.float(),
        }
        if self.uncertainty_estimation:
            out_dict.update({
                'AUSE_AEPE': self.AUSE_AEPE / self.nbr_samples.double(),
            })
        return out_dict

    def compute_aucs(self, gt, pred, uncert, intervals=50):
        """
        Computation of sparsification curve, oracle curve and auc metric (area below the difference of the two curves),
        for each metrics (AEPE, PCK ..).
        Args:
            gt: gt flow field, shape #number elements, 2
            pred: predicted flow field, shape #number elements, 2
            uncert: predicted uncertainty measure, shape #number elements
            intervals: number of intervals to compute the sparsification plot
        Returns:
            dictionary with sparsification, oracle and AUC for each metric (here EPE, PCK1 and PCK5).
        """
        # uncertainty_metrics = ['EPE', 'PCK1', 'PCK5']
        uncertainty_metrics = ['EPE']
        value_for_no_pixels = {'EPE': 0.0, 'PCK1': 1.0, 'PCK5': 1.0}
        # results dictionaries
        AUSE = {'EPE': 0, 'PCK1': 0, 'PCK5': 0}

        # revert order (high uncertainty first)
        uncert = -uncert  # shape #number_elements

        # list the EPE, as the uncertainty. negative because we want high uncertainty first when taking percentile!
        true_uncert = - torch.linalg.norm(gt - pred, ord=2, dim=1)

        # prepare subsets for sampling and for area computation
        quants = [1. / intervals * t for t in range(0, intervals)]
        plotx = torch.tensor(
            [1. / intervals * t for t in range(0, intervals + 1)], device=gt.device)

        # get percentiles for sampling and corresponding subsets
        thresholds = [torch.quantile(uncert.float(), q) for q in quants]
        subs = [(uncert.ge(t)) for t in thresholds]

        # compute sparsification curves for each metric (add 0 for final sampling)
        # calculates the metrics for each interval
        sparse_curve = {
            m: torch.stack([self.compute_eigen_errors_v2(gt, pred, metrics=[m], mask=sub, reduce_mean=True)[0] for sub in subs] +
                           [torch.tensor(value_for_no_pixels[m], device=gt.device)]) for m in uncertainty_metrics}

        # human-readable call
        '''
        sparse_curve =  {"rmse":[compute_eigen_errors_v2(gt,pred,metrics=["rmse"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0], 
                        "a1":[compute_eigen_errors_v2(gt,pred,metrics=["a1"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0],
                        "abs_rel":[compute_eigen_errors_v2(gt,pred,metrics=["abs_rel"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0]}
        '''

        # get percentiles for optimal sampling and corresponding subsets (based on real EPE)
        opt_thresholds = [torch.quantile(
            true_uncert.float(), q) for q in quants]
        opt_subs = [(true_uncert.ge(o)) for o in opt_thresholds]

        # compute sparsification curves for optimal sampling (add 0 for final sampling)
        opt_curve = {m: torch.stack([self.compute_eigen_errors_v2(gt, pred, metrics=[m], mask=opt_sub, reduce_mean=True)[0] for opt_sub in
                                     opt_subs] + [torch.tensor(value_for_no_pixels[m], device=gt.device)]) for m in uncertainty_metrics}

        # compute error and gain metrics
        for m in uncertainty_metrics:
            mmax = opt_curve[m].max() + 1e-6
            # normalize both to 0-1 first
            opt_curve[m] = opt_curve[m] / mmax
            sparse_curve[m] = sparse_curve[m] / mmax

            # error: subtract from method sparsification (first term) the oracle sparsification (second term)
            AUSE[m] = torch.abs(torch.trapz(sparse_curve[m], x=plotx) -
                                torch.trapz(opt_curve[m], x=plotx))

        return AUSE

    def compute_eigen_errors_v2(self, gt, pred, metrics=['EPE', 'PCK1', 'PCK5'], mask=None, reduce_mean=True):
        """Revised compute_eigen_errors function used for uncertainty metrics, with optional reduce_mean argument and (1-a1) computation
        """
        results = []

        # in shape (#number_elements, 2)
        # mask shape #number_of_elements
        if mask is not None:
            pred = pred[mask]
            gt = gt[mask]

        if "EPE" in metrics:
            epe = torch.linalg.norm(gt - pred, ord=2, dim=1)
            if reduce_mean:
                epe = epe.mean()
            results.append(epe)

        if "PCK1" in metrics:
            if pred.shape[0] == 0:
                pck1 = pred.new_zeros([])
            else:
                px_1 = self.correct_correspondences(
                    pred, gt, alpha=1.0, img_size=1.0)
                pck1 = px_1 / (pred.shape[0])
            results.append(pck1)

        if "PCK5" in metrics:
            if pred.shape[0] == 0:
                pck5 = pred.new_zeros([])
            else:
                px_5 = self.correct_correspondences(
                    pred, gt, alpha=5.0, img_size=1.0)
                pck5 = px_5 / (pred.shape[0])
            results.append(pck5)

        return results

    @staticmethod
    def correct_correspondences(input_flow, target_flow, alpha, img_size, epe_tensor=None):
        """
        Computation PCK, i.e number of the pixels within a certain threshold
        Args:
            input_flow: estimated flow [BxHxW,2]
            target_flow: ground-truth flow [BxHxW,2]
            alpha: threshold
            img_size: image load_size
            epe_tensor: epe tensor already computed [BxHxW, 1], default is None
        Output:
            PCK metric
        """
        if epe_tensor is not None:
            dist = epe_tensor
        else:
            dist = torch.linalg.norm(target_flow - input_flow, ord=2, dim=1)
        # dist is shape BxHgtxWgt
        pck_threshold = alpha * img_size
        # Computes dist ≤ pck_threshold element-wise (element then equal to 1)
        mask = dist.le(pck_threshold)
        return mask.sum()


class IoU(torchmetrics.JaccardIndex):
    """ Wrapper because native IoU does not support ignore_index. 
    https://github.com/PyTorchLightning/metrics/issues/304
    """

    def __init__(self, over_present_classes: bool = False, **kwargs):
        self.over_present_classes = over_present_classes
        super().__init__(**kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        target = target.view(-1)
        N = len(target)
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        if len(preds.shape) == 4:
            C = preds.shape[1]
            preds = preds.permute(0, 2, 3, 1).view(N, C)
            preds = preds[valid_mask, :]
        elif len(preds.shape) == 3:
            preds = preds.view(N)
            preds = preds[valid_mask]
        confmat = _confusion_matrix_update(
            preds, target, self.num_classes, self.threshold, self.multilabel)
        self.confmat += confmat

    def compute(self) -> Tensor:
        """Computes intersection over union (IoU)"""
        return self._jaccard_from_confmat(
            self.confmat,
            self.num_classes,
            self.average,
            None,
            self.absent_score,
            self.over_present_classes,
        )

    def _jaccard_from_confmat(
        self,
        confmat: Tensor,
        num_classes: int,
        average: Optional[str] = "macro",
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        over_present_classes: bool = False,
    ) -> Tensor:
        """Computes the intersection over union from confusion matrix.
        Args:
            confmat: Confusion matrix without normalization
            num_classes: Number of classes for a given prediction and target tensor
            average:
                Defines the reduction that is applied. Should be one of the following:
                - ``'macro'`` [default]: Calculate the metric for each class separately, and average the
                metrics across classes (with equal weights for each class).
                - ``'micro'``: Calculate the metric globally, across all samples and classes.
                - ``'weighted'``: Calculate the metric for each class separately, and average the
                metrics across classes, weighting each class by its support (``tp + fn``).
                - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
                the metric for every class. Note that if a given class doesn't occur in the
                `preds` or `target`, the value for the class will be ``nan``.
            ignore_index: optional int specifying a target class to ignore. If given, this class index does not contribute
                to the returned score, regardless of reduction method.
            absent_score: score to use for an individual class, if no instances of the class index were present in `pred`
                AND no instances of the class index were present in `target`.
        """
        allowed_average = ["macro", "weighted", "none", None]
        if average not in allowed_average:
            raise ValueError(
                f"The `average` has to be one of {allowed_average}, got {average}.")

        # Remove the ignored class index from the scores.
        if ignore_index is not None and 0 <= ignore_index < num_classes:
            confmat[ignore_index] = 0.0

        if average == "none" or average is None:
            intersection = torch.diag(confmat)
            union = confmat.sum(0) + confmat.sum(1) - intersection

            present_classes = confmat.sum(dim=1) != 0

            # If this class is absent in both target AND pred (union == 0), then use the absent_score for this class.
            scores = intersection.float() / union.float()
            scores[union == 0] = absent_score

            if ignore_index is not None and 0 <= ignore_index < num_classes:
                scores = torch.cat(
                    [
                        scores[:ignore_index],
                        scores[ignore_index + 1:],
                    ]
                )
                present_classes = torch.cat(
                    [
                        present_classes[:ignore_index],
                        present_classes[ignore_index + 1:],
                    ]
                )

            if over_present_classes:
                scores = scores[present_classes]

            return scores

        if average == "macro":
            scores = self._jaccard_from_confmat(
                confmat, num_classes, average="none", ignore_index=ignore_index,
                absent_score=absent_score, over_present_classes=over_present_classes
            )
            return torch.mean(scores)

        if average == "micro":
            raise NotImplementedError()

        weights = torch.sum(confmat, dim=1).float() / \
            torch.sum(confmat).float()
        scores = self._jaccard_from_confmat(
            confmat, num_classes, average="none", ignore_index=ignore_index,
            absent_score=absent_score, over_present_classes=over_present_classes
        )
        return torch.sum(weights * scores)
