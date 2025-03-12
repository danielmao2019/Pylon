import torch
from criteria.wrappers import SingleTaskCriterion
from criteria.vision_2d.dense_prediction.dense_classification.semantic_segmentation import SemanticSegmentationCriterion
from criteria.vision_2d.dense_prediction.dense_classification.dice_loss import DiceLoss


class CEDiceLoss(SingleTaskCriterion):
    """
    Combined Cross-Entropy and Dice Loss for semantic segmentation tasks.
    
    This loss combines the standard Cross-Entropy loss with the Dice loss
    to benefit from both:
    - Cross-Entropy provides good gradients for accurate per-pixel classification
    - Dice loss handles class imbalance well and optimizes the IoU metric
    
    This implementation supports:
    - Combined CE and Dice loss (CE + Dice)
    - Class weights to handle class imbalance
    - Ignore value to exclude specific pixel values from loss computation
    
    Attributes:
        ignore_value (int): Value to ignore in loss computation
        reduction (str): How to reduce the loss over the batch dimension ('mean' or 'sum')
        class_weights (Optional[torch.Tensor]): Optional weights for each class
    """

    COMBINE_OPTIONS = {'mean', 'sum'}

    def __init__(self, combine='sum', class_weights=None, ignore_value=255) -> None:
        super(CEDiceLoss, self).__init__()
        assert combine in self.COMBINE_OPTIONS
        self.ce_loss = SemanticSegmentationCriterion(reduction='mean', class_weights=class_weights, ignore_value=ignore_value)
        self.dice_loss = DiceLoss(reduction='mean', class_weights=class_weights, ignore_value=ignore_value)
        self.combine = combine

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(y_pred=y_pred, y_true=y_true)
        dice_loss = self.dice_loss(y_pred=y_pred, y_true=y_true)
        total_loss = ce_loss + dice_loss
        if self.combine == 'mean':
            total_loss = total_loss / 2
        return total_loss
