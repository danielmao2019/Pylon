from typing import Dict
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.point_cloud_ops.apply_transform import apply_transform


class InlierRatio(SingleTaskMetric):
    """
    Inlier Ratio metric for 3D point cloud registration.

    Inlier Ratio measures the proportion of points in the predicted point cloud
    that are within a certain distance threshold of their nearest neighbors in
    the target point cloud.
    """

    DIRECTION = +1  # Higher is better

    def __init__(self, threshold: float) -> None:
        """
        Initialize the Inlier Ratio metric.

        Args:
            threshold: Distance threshold for considering a point as an inlier
        """
        super(InlierRatio, self).__init__()
        self.threshold = threshold

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert isinstance(y_pred, dict), f"y_pred must be a dictionary. Got {type(y_pred)}."
        assert y_pred.keys() == {'src_points', 'tgt_points'}, f"{y_pred.keys()=}"
        assert isinstance(y_pred['src_points'], torch.Tensor), f"src_points must be a tensor. Got {type(y_pred['src_points'])}."
        assert y_pred['src_points'].ndim == 2 and y_pred['src_points'].shape[1] == 3, f"{y_pred['src_points'].shape=}"
        assert isinstance(y_pred['tgt_points'], torch.Tensor), f"tgt_points must be a tensor. Got {type(y_pred['tgt_points'])}."
        assert y_pred['tgt_points'].ndim == 2 and y_pred['tgt_points'].shape[1] == 3, f"{y_pred['tgt_points'].shape=}"
        assert len(y_pred['src_points']) == len(y_pred['tgt_points']), f"{len(y_pred['src_points'])=}, {len(y_pred['tgt_points'])=}"
        assert isinstance(y_true, dict), f"y_true must be a dictionary. Got {type(y_true)}."
        assert y_true.keys() == {'transform'}, f"{y_true.keys()=}"
        assert y_true['transform'].shape == (1, 4, 4), f"{y_true['transform'].shape=}"

        src_points = y_pred['src_points']
        tgt_points = y_pred['tgt_points']
        transform = y_true['transform'].squeeze(0)

        src_points = apply_transform(src_points, transform)
        distances = torch.linalg.norm(src_points - tgt_points, dim=1)
        inlier_mask = (distances <= self.threshold)
        inlier_ratio = torch.mean(inlier_mask.float())
        scores = {'inlier_ratio': inlier_ratio}
        self.add_to_buffer(scores)
        return scores
