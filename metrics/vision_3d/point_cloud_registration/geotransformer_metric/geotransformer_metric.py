from typing import Dict, Tuple
import torch
from easydict import EasyDict
from models.point_cloud_registration.geotransformer.transformation import apply_transform
from metrics.vision_3d.point_cloud_registration.geotransformer_metric.metrics import isotropic_transform_error
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.ops import apply_tensor_op


class GeoTransformerMetric(SingleTaskMetric):
    def __init__(self, **cfg):
        super(GeoTransformerMetric, self).__init__()
        cfg = EasyDict(cfg)
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse(self, y_pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref_length_c = y_pred['ref_points_c'].shape[0]
        src_length_c = y_pred['src_points_c'].shape[0]
        gt_node_corr_overlaps = y_pred['gt_node_corr_overlaps']
        gt_node_corr_indices = y_pred['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = y_pred['ref_node_corr_indices']
        src_node_corr_indices = y_pred['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        transform = y_true['transform']
        assert transform.shape == (1, 4, 4), f"{transform.shape=}"
        transform = transform.squeeze(0)
        ref_corr_points = y_pred['ref_corr_points']
        src_corr_points = y_pred['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transform = y_true['transform']
        assert transform.shape == (1, 4, 4), f"{transform.shape=}"
        transform = transform.squeeze(0)

        est_transform = y_pred['estimated_transform']
        assert est_transform.shape == (1, 4, 4), f"{est_transform.shape=}"

        src_points = y_pred['src_points']

        rre, rte = isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c_precision = self.evaluate_coarse(y_pred)
        f_precision = self.evaluate_fine(y_pred, y_true)
        rre, rte, rmse, recall = self.evaluate_registration(y_pred, y_true)
        score: Dict[str, torch.Tensor] = {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }
        score = apply_tensor_op(func=lambda x: x.detach().cpu(), inputs=score)
        # log score
        self.buffer.append(score)
        return score
