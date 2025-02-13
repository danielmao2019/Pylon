import torch
from criteria.vision_2d import SemanticMapBaseCriterion


class DiceLoss(SemanticMapBaseCriterion):

    def _compute_semantic_map_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        inter = torch.sum(y_pred * y_true, dim=(2, 3))
        total = torch.sum(y_pred + y_true, dim=(2, 3))
        dice_loss = 1 - 2 * inter / total
        return dice_loss
