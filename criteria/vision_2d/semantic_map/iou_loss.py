import torch
from criteria.vision_2d import SemanticMapBaseCriterion


class IoULoss(SemanticMapBaseCriterion):

    def _compute_semantic_map_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        inter = (y_true * y_pred).sum(dim=[2, 3])
        union = y_true.sum(dim=[2, 3]) + y_pred.sum(dim=[2, 3]) - inter
        iou_loss = inter / union
        return iou_loss
