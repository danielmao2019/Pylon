from typing import Tuple, Dict
from easydict import EasyDict
import torch
from metrics.vision_3d.point_cloud_registration.inlier_ratio import InlierRatio
from metrics.vision_3d.point_cloud_registration.isotropic_transform_error import IsotropicTransformError
from metrics.vision_3d.point_cloud_registration.point_inlier_ratio import PointInlierRatio
from metrics.wrappers.single_task_metric import SingleTaskMetric
from models.point_cloud_registration.geotransformer.transformation import apply_transform


class GeoTransformerMetric(SingleTaskMetric):

    def __init__(self, **cfg):
        cfg = EasyDict(cfg)
        super(GeoTransformerMetric, self).__init__()
        self.acceptance_overlap = cfg.acceptance_overlap
        self.acceptance_radius = cfg.acceptance_radius
        self.acceptance_rmse = cfg.rmse_threshold
        self.transform_metric = IsotropicTransformError(use_buffer=False)

    @torch.no_grad()
    def evaluate_coarse(self, y_pred: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        assert isinstance(y_pred, dict), f"Expected dict for y_pred, got {type(y_pred)}"
        assert y_pred.keys() >= {'src_points_c', 'ref_points_c', 'src_node_corr_indices', 'ref_node_corr_indices', 'gt_node_corr_indices', 'gt_node_corr_overlaps'}, f"{y_pred.keys()=}"

        gt_node_corr_overlaps = y_pred['gt_node_corr_overlaps']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = y_pred['gt_node_corr_indices']
        gt_node_corr_indices = gt_node_corr_indices[masks]

        point_inlier_ratio = PointInlierRatio(use_buffer=False)(
            y_pred={
                'src_points': y_pred['src_points_c'],
                'tgt_points': y_pred['ref_points_c'],
                'correspondences': torch.stack([y_pred['src_node_corr_indices'], y_pred['ref_node_corr_indices']], dim=1),
            },
            y_true={
                'correspondences': gt_node_corr_indices,
            },
            idx=idx,
        )
        assert point_inlier_ratio.keys() == {'point_inlier_ratio'}, f"{point_inlier_ratio.keys()=}"
        return point_inlier_ratio['point_inlier_ratio']

    @torch.no_grad()
    def evaluate_fine(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        inlier_ratio = InlierRatio(threshold=self.acceptance_radius, use_buffer=False)(
            y_pred={
                'src_points': y_pred['src_corr_points'],
                'tgt_points': y_pred['ref_corr_points'],
            },
            y_true={
                'transform': y_true['transform'],
            },
            idx=idx,
        )
        assert inlier_ratio.keys() == {'inlier_ratio'}, f"{inlier_ratio.keys()=}"
        return inlier_ratio['inlier_ratio']

    @torch.no_grad()
    def evaluate_registration(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor], idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transform = y_true['transform']
        assert transform.shape == (1, 4, 4), f"{transform.shape=}"
        transform = transform.squeeze(0)

        est_transform = y_pred['estimated_transform']
        assert est_transform.shape == (1, 4, 4), f"{est_transform.shape=}"
        est_transform = est_transform.squeeze(0)

        src_points = y_pred['src_points']

        # Use IsotropicTransformError metric
        transform_scores = self.transform_metric(
            y_pred={'transform': est_transform.unsqueeze(0)},
            y_true={'transform': transform.unsqueeze(0)},
            idx=idx,
        )
        rre = transform_scores['rotation_error']
        rte = transform_scores['translation_error']

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor], idx: int) -> Dict[str, torch.Tensor]:
        c_precision = self.evaluate_coarse(y_pred, idx)
        f_precision = self.evaluate_fine(y_pred, y_true, idx)
        rre, rte, rmse, recall = self.evaluate_registration(y_pred, y_true, idx)
        scores: Dict[str, torch.Tensor] = {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }
        self.add_to_buffer(scores, idx)
        return scores
