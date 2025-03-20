import pytest
import torch
from models.change_detection.cdmaskformer import CDMaskFormer, CDMaskFormerBackbone, CDMaskFormerHead


@pytest.mark.parametrize('mode', ['train', 'eval'])
def test_cdmaskformer(mode):
    model = CDMaskFormer(
        backbone_cfg={
            'class': CDMaskFormerBackbone,
            'args': {},
        },
        head_cfg={
            'class': CDMaskFormerHead,
            'args': {
                'channels': [64, 128, 192, 256],
                'num_classes': 1,
                'num_queries': 5,
                'dec_layers': 14,
            },
        }
    )
    if mode == 'train':
        model.train()
    else:
        model.eval()

    inputs = {
        'img_1': torch.randn(1, 3, 256, 256),
        'img_2': torch.randn(1, 3, 256, 256),
    }
    outputs = model.forward(inputs)

    if mode == 'train':
        assert isinstance(outputs, dict)
        assert outputs.keys() == {'pred_logits', 'pred_masks'}
        assert outputs['pred_logits'].shape == (1, 5, 56, 56), f"{outputs['pred_logits'].shape}"
        assert outputs['pred_masks'].shape == (1, 5, 56, 56), f"{outputs['pred_masks'].shape}"
    else:
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (1, 56, 56), f"{outputs.shape}"
