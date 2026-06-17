"""
MODELS.MULTI_TASK_LEARNING.BACKBONES API
"""

from models.multi_task_learning.backbones.lenet5 import LeNet5
from models.multi_task_learning.backbones.resnet.resnet import (
    ResNet50Dilated,
    resnet18,
    resnet50,
)
from models.multi_task_learning.backbones.segnet.segnet import SegNet
from models.multi_task_learning.backbones.unet.unet_decoder import UNetDecoder
from models.multi_task_learning.backbones.unet.unet_encoder import UNetEncoder

__all__ = (
    'LeNet5',
    'resnet18',
    'resnet50',
    'ResNet50Dilated',
    'SegNet',
    'UNetEncoder',
    'UNetDecoder',
)
