import torch
from configs.common.models.change_detection.oneconvfusionkpconv import model_config
from utils.builders import build_model_from_config


def test_oneconvfusionkpconv():
    model = build_model_from_config(model_config)
    inputs = {
        'pc_1': {
            'pos': torch.randn(1024, 3),
            'feat': torch.randn(1024, 3),
            'batch': torch.randint(0, 2, (1024,)),
        },
        'pc_2': {
            'pos': torch.randn(1024, 3),
            'feat': torch.randn(1024, 3),
            'batch': torch.randint(0, 2, (1024,)),
        },
    }
    outputs = model(inputs)
    assert outputs.shape == (1024, 1024)
