import torch
import models
from utils.builders.builder import build_from_config


def test_cdmaskformer():
    # Create model config following the Pylon builder pattern
    cfg = {
        'class': models.change_detection.cdmaskformer.models.build_model.CDMaskFormer,
        'args': {
            'backbone': {
                'class': models.change_detection.cdmaskformer.backbones.base.Base,
                'args': {
                    'name': 'Seaformer'
                },
            },
            'decoderhead': {
                'class': models.change_detection.cdmaskformer.decoder_heads.cdmask.CDMask,
                'args': {
                    'channels': [64, 128, 192, 256],
                    'num_classes': 1,  # num_class - 1 from original config
                    'num_queries': 5,
                    'dec_layers': 14
                },
            },
        },
    }

    # Initialize model using the builder
    model = build_from_config(cfg)

    # Create dummy inputs with batch dimension
    batch_size = 2
    num_channels = 3
    height = 256
    width = 256
    
    # Create dummy inputs following the data structure from dataloader
    inputs = {
        'img_1': torch.randn(batch_size, num_channels, height, width),
        'img_2': torch.randn(batch_size, num_channels, height, width)
    }

    # Run model forward pass
    outputs = model(inputs)

    # Check output structure
    assert isinstance(outputs, list)
    assert len(outputs) == 2  # [pred_masks, outputs]

    # Check pred_masks shape
    pred_masks = outputs[0]
    assert isinstance(pred_masks, torch.Tensor)
    assert pred_masks.shape == (batch_size, 1, height, width)  # 1 class for change detection

    # Check outputs dictionary structure
    outputs_dict = outputs[1]
    assert isinstance(outputs_dict, dict)
    assert 'pred_logits' in outputs_dict
    assert 'pred_masks' in outputs_dict
    assert 'aux_outputs' in outputs_dict

    # Check shapes of outputs
    assert outputs_dict['pred_logits'].shape == (batch_size, 5, 2)  # [B, num_queries, num_classes+1]
    assert outputs_dict['pred_masks'].shape == (batch_size, 5, height//4, width//4)  # [B, num_queries, H/4, W/4]
    assert len(outputs_dict['aux_outputs']) == 13  # dec_layers - 1

    # Check aux_outputs structure
    for aux_output in outputs_dict['aux_outputs']:
        assert isinstance(aux_output, dict)
        assert 'pred_logits' in aux_output
        assert 'pred_masks' in aux_output
        assert aux_output['pred_logits'].shape == (batch_size, 5, 2)
        assert aux_output['pred_masks'].shape == (batch_size, 5, height//4, width//4)
