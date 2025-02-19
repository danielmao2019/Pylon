import pytest
from .a2net_lwganet import LWGANet
import torch

def test_snunet() -> None:
    model = LWGANet()
    inputs = {
        'img_1': torch.zeros(size=(16, 3, 32, 32)),
        'img_2': torch.zeros(size=(16, 3, 32, 32)),
    }
    output = model(inputs)
    assert output.shape == torch.Size([16, 2, 32, 32]), f'{output.shape=}'