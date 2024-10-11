"""
MODELS.BACKBONES API
"""
from models.backbones.resnet.resnet import resnet18, resnet50, ResNet50Dilated
from models.backbones.segnet.segnet import SegNet


__all__ = (
    "resnet18",
    "resnet50",
    "ResNet50Dilated",
    "SegNet",
)
