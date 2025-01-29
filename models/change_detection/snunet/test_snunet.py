"""Reference: https://pytorch.org/vision/0.11/feature_extraction.html
"""
import pytest
from .snunet import SNUNet_ECAM
import torch

def test_snunet() -> None:
    model = SNUNet_ECAM()
    inputs = {
        'img_1': torch.zeros(size=(16, 3, 32, 32)),
        'img_2': torch.zeros(size=(16, 3, 32, 32)),
    }
    model(inputs)
