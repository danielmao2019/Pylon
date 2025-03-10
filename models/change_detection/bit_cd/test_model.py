import pytest
from .model import BASE_Transformer
import torch

def test_snunet() -> None:
    model = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, with_pos='learned', enc_depth=1, dec_depth=8)
    inputs = {
        'img_1': torch.zeros(size=(16, 3, 32, 32)),
        'img_2': torch.zeros(size=(16, 3, 32, 32)),
    }
    output = model(inputs)
    assert output.shape == torch.Size([16, 2, 32, 32]), f'{output.shape=}'