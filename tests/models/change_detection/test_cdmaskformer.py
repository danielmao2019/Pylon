import pytest
import torch
from models.change_detection.cdmaskformer.models.cdmaskformer import CDMaskFormer

def test_cdmaskformer() -> None:
    # Create model with test configuration
    backbone_name = 'resnet50'
    backbone_args = {'pretrained': False}
    cdmask_args = {
        'num_classes': 1,
        'num_queries': 10,
        'dec_layers': 6
    }
    
    # Initialize model
    model = CDMaskFormer(backbone_name, backbone_args, cdmask_args)
    
    # Create test inputs
    inputs = {
        'img_1': torch.zeros(size=(2, 3, 256, 256)),
        'img_2': torch.zeros(size=(2, 3, 256, 256)),
    }
    
    # Test forward pass
    model.eval()  # Set to eval mode to get the final change map output
    output = model(inputs)
    
    # Since in eval mode, we should get a tensor with shape [B, C, H, W]
    # where C is the number of classes (1 for binary change detection)
    assert output.shape == torch.Size([2, 1, 64, 64]), f'{output.shape=}'
    
    # Test training mode
    model.train()
    outputs = model(inputs)
    
    # In training mode, we should get a dictionary with pred_logits and pred_masks
    assert isinstance(outputs, dict), "Training output should be a dictionary"
    assert "pred_logits" in outputs, "Missing pred_logits in output dictionary"
    assert "pred_masks" in outputs, "Missing pred_masks in output dictionary"
    
    # Check output shapes
    assert outputs["pred_logits"].shape[0] == 2, "Batch size mismatch in pred_logits"
    assert outputs["pred_masks"].shape[0] == 2, "Batch size mismatch in pred_masks"
    assert outputs["pred_masks"].shape[-2:] == (64, 64), "Spatial dimensions mismatch in pred_masks" 