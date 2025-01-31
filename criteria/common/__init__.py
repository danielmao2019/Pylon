"""
CRITERIA.COMMON API
"""
from criteria.common.focal_loss import FocalLoss
from criteria.common.binary_dice_loss import BinaryDiceLoss
from criteria.common.dice_loss import DiceLoss


__all__ = (
    'FocalLoss',
    'BinaryDiceLoss',
    'DiceLoss'
)
