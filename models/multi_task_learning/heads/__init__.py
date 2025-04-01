"""
MODELS.MULTI_TASK_LEARNING.HEADS API
"""
from models.multi_task_learning.heads.ppm_decoder import PyramidPoolingModule
from models.multi_task_learning.heads.two_conv_decoder import TwoConvDecoder


__all__ = (
    'PyramidPoolingModule',
    'TwoConvDecoder',
)
