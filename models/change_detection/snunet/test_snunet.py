import pytest
from .snunet import SNUNet_ECAM
import torch

def test_snunet() -> None:
    model = SNUNet_ECAM()
    inputs = {
        'img_1': torch.zeros(size=(16, 3, 32, 32)),
        'img_2': torch.zeros(size=(16, 3, 32, 32)),
    }
    output = model(inputs)
    assert output.shape == torch.Size([16, 2, 32, 32]), f'{output.shape=}'
