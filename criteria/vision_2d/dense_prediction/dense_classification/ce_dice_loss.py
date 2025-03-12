import torch
from criteria.wrappers import SingleTaskCriterion
from criteria.vision_2d.dense_prediction.dense_classification.semantic_segmentation import SemanticSegmentationCriterion
from criteria.vision_2d.dense_prediction.dense_classification.dice_loss import DiceLoss


class CEDiceLoss(SingleTaskCriterion):

    COMBINE_OPTIONS = {'mean', 'sum'}

    def __init__(self, combine='sum', class_weights=None, ignore_index=255) -> None:
        super(CEDiceLoss, self).__init__()
        assert combine in self.COMBINE_OPTIONS
        self.ce_loss = SemanticSegmentationCriterion(reduction='mean', class_weights=class_weights, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(reduction='mean', class_weights=class_weights, ignore_index=ignore_index)
        self.combine = combine

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(y_pred=y_pred, y_true=y_true)
        dice_loss = self.dice_loss(y_pred=y_pred, y_true=y_true)
        total_loss = ce_loss + dice_loss
        if self.combine == 'mean':
            total_loss = total_loss / 2
        return total_loss
