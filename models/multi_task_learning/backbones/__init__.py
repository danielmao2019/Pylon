"""
MODELS.MULTI_TASK_LEARNING.BACKBONES API
"""
from models.multi_task_learning.backbones.lenet5 import LeNet5
from models.multi_task_learning.backbones.resnet.resnet import resnet18, resnet50, ResNet50Dilated
from models.multi_task_learning.backbones.segnet.segnet import SegNet
from models.multi_task_learning.backbones.unet.unet_encoder import UNetEncoder
from models.multi_task_learning.backbones.unet.unet_decoder import UNetDecoder


__all__ = (
    'LeNet5',
    'resnet18',
    'resnet50',
    'ResNet50Dilated',
    'SegNet',
    'UNetEncoder',
    'UNetDecoder',
)
