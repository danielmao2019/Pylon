"""
MODELS.BACKBONES API
"""
from models.backbones.lenet5 import LeNet5
from models.backbones.resnet.resnet import resnet18, resnet50, ResNet50Dilated
from models.backbones.segnet.segnet import SegNet
from models.backbones.unet.unet_encoder import UNetEncoder
from models.backbones.unet.unet_decoder import UNetDecoder


__all__ = (
    'LeNet5',
    'resnet18',
    'resnet50',
    'ResNet50Dilated',
    'SegNet',
    'UNetEncoder',
    'UNetDecoder',
)
