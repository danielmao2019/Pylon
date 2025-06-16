from typing import Dict
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric


class PointInlierRatio(SingleTaskMetric):

    DIRECTION = +1  # Higher is better

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert isinstance(y_pred, dict), f"Expected dict for y_pred, got {type(y_pred)}"
        assert y_pred.keys() == {'src_points', 'tgt_points', 'correspondences'}, f"{y_pred.keys()=}"
        assert isinstance(y_pred['src_points'], torch.Tensor), f"Expected torch.Tensor for src_points, got {type(y_pred['src_points'])}"
        assert isinstance(y_pred['tgt_points'], torch.Tensor), f"Expected torch.Tensor for tgt_points, got {type(y_pred['tgt_points'])}"
        assert y_pred['src_points'].ndim == 2 and y_pred['src_points'].shape[1] == 3, f"{y_pred['src_points'].shape=}"
        assert y_pred['tgt_points'].ndim == 2 and y_pred['tgt_points'].shape[1] == 3, f"{y_pred['tgt_points'].shape=}"
        assert isinstance(y_pred['correspondences'], torch.Tensor), f"Expected torch.Tensor for correspondences, got {type(y_pred['correspondences'])}"
        assert y_pred['correspondences'].ndim == 2 and y_pred['correspondences'].shape[1] == 2, f"{y_pred['correspondences'].shape=}"
        assert isinstance(y_true, dict), f"Expected dict for y_true, got {type(y_true)}"
        assert y_true.keys() >= {'correspondences'}, f"{y_true.keys()=}"
        assert isinstance(y_true['correspondences'], torch.Tensor), f"Expected torch.Tensor for correspondences, got {type(y_true['correspondences'])}"
        assert y_true['correspondences'].ndim == 2 and y_true['correspondences'].shape[1] == 2, f"{y_true['correspondences'].shape=}"

        device = y_pred['src_points'].device
        src_length = y_pred['src_points'].shape[0]
        tgt_length = y_pred['tgt_points'].shape[0]
        corr_map = torch.zeros(size=(src_length, tgt_length), device=device)
        corr_map[y_true['correspondences'][:, 0], y_true['correspondences'][:, 1]] = 1.0
        point_inlier_ratio = corr_map[y_pred['correspondences'][:, 0], y_pred['correspondences'][:, 1]].mean()
        scores = {'point_inlier_ratio': point_inlier_ratio}
        self.add_to_buffer(scores)
        return scores
