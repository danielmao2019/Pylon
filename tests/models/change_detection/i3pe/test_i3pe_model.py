from models.change_detection.i3pe.i3pe_model import I3PEModel
import torch


def test_i3pe_model() -> None:
    model = I3PEModel().to('cuda')
    inputs = {
        'img_1': torch.zeros(size=(4, 3, 224, 224), device='cuda'),
        'img_2': torch.zeros(size=(4, 3, 224, 224), device='cuda'),
    }
    outputs = model(inputs)
    assert type(outputs) == dict and set(outputs.keys()) == {'change_map_12', 'change_map_21'}
    assert all(x.shape == (4, 2, 224, 224) for x in outputs.values())
