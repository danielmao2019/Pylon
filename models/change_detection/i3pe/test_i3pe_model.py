from .i3pe_model import I3PEModel
import torch


def test_i3pe_model() -> None:
    model = I3PEModel().to('cuda')
    img_1 = torch.zeros(size=(1, 3, 224, 224), device='cuda')
    img_2 = torch.zeros(size=(1, 3, 224, 224), device='cuda')
    _ = model(img_1, img_2)
