from criteria.wrappers.hybrid_criterion import HybridCriterion
from criteria.common.focal_loss import FocalLoss
from criteria.vision_2d.dense_prediction.dense_classification.dice_loss import DiceLoss


class FocalDiceLoss(HybridCriterion):
    """
    Combined Focal and Dice Loss for semantic segmentation tasks.
    
    This loss combines the Focal Loss with the Dice loss to benefit from both:
    - Focal Loss helps handle class imbalance by down-weighting easy examples
    - Dice loss optimizes the IoU metric and provides good gradients for imbalanced classes
    
    This implementation supports:
    - Combined Focal and Dice loss (Focal + Dice)
    - Class weights to handle class imbalance
    - Ignore value to exclude specific pixel values from loss computation
    - Gamma parameter for Focal Loss to control the focusing effect
    
    Attributes:
        reduction (str): How to reduce the loss over the batch dimension ('mean' or 'sum')
        class_weights (Optional[torch.Tensor]): Optional weights for each class
        ignore_value (int): Value to ignore in loss computation
        gamma (float): Focusing parameter for Focal Loss (default: 0.0)
    """


    def __init__(self, combine='sum', class_weights=None, ignore_value=255, gamma=0.0) -> None:
        criteria_cfg = [
            {
                'class': FocalLoss,
                'args': {
                    'gamma': gamma,
                    'class_weights': class_weights,
                    'ignore_value': ignore_value,
                },
            },
            {
                'class': DiceLoss,
                'args': {
                    'reduction': 'mean',
                    'class_weights': class_weights,
                    'ignore_value': ignore_value,
                },
            },
        ]
        super(FocalDiceLoss, self).__init__(combine=combine, criteria_cfg=criteria_cfg)
