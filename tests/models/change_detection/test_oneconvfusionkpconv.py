from models.change_detection.siamese_kpconv_variants.oneconvfusionkpconv import OneConvFusionKPConv
import torch


def test_oneconvfusionkpconv():
    model = OneConvFusionKPConv()
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
