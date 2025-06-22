from criteria.vision_2d.dense_prediction.dense_classification.dice_loss import DiceLoss
from criteria.vision_2d.dense_prediction.dense_classification.semantic_segmentation import SemanticSegmentationCriterion
from criteria.wrappers.hybrid_criterion import HybridCriterion


class CEDiceLoss(HybridCriterion):
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

    def __init__(self, combine='sum', class_weights=None, ignore_value=255, **kwargs) -> None:
        criteria_cfg = [
            {
                'class': SemanticSegmentationCriterion,
                'args': {
                    'reduction': 'mean',
                    'class_weights': class_weights,
                    'ignore_value': ignore_value
                }
            },
            {
                'class': DiceLoss,
                'args': {
                    'reduction': 'mean',
                    'class_weights': class_weights,
                    'ignore_value': ignore_value
                },
            },
        ]
        super(CEDiceLoss, self).__init__(combine=combine, criteria_cfg=criteria_cfg, **kwargs)
