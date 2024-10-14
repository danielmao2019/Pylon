import torch
from criteria.wrappers.single_task_criterion import SingleTaskCriterion
from utils.input_checks import check_depth_estimation


class DepthEstimationCriterion(SingleTaskCriterion):

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # input checks
        check_depth_estimation(y_pred=y_pred, y_true=y_true)
        y_pred = y_pred.squeeze(1)
        # compute loss
        mask = y_true != 0
        assert mask.sum() >= 1
        return torch.nn.functional.l1_loss(y_pred[mask], y_true[mask], reduction='mean')
