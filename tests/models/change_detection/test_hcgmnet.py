import pytest
import torch
from models.change_detection.hcgmnet import HCGMNet

def test_hcgmnet_inference() -> None:
    """Test HCGMNet forward pass during inference."""
    model = HCGMNet(num_classes=2)
    model.eval()  # Set to evaluation mode
    
    # Create dummy input tensors
    inputs = {
        'img_1': torch.zeros(size=(4, 3, 224, 224)),
        'img_2': torch.zeros(size=(4, 3, 224, 224)),
    }
    
    # Run forward pass
    outputs = model(inputs)
    
    # Check output structure and shapes
    assert isinstance(outputs, dict), f"Expected dict output, got {type(outputs)}"
    assert 'change_map' in outputs, f"Expected 'change_map' key in outputs, got {outputs.keys()}"
    assert outputs['change_map'].shape == torch.Size([4, 2, 224, 224]), f"Unexpected shape: {outputs['change_map'].shape}"
    assert 'aux_change_map' not in outputs, f"'aux_change_map' should not be present in inference mode"

def test_hcgmnet_training() -> None:
    """Test HCGMNet forward pass during training."""
    model = HCGMNet(num_classes=2)
    model.train()  # Set to training mode
    
    # Create dummy input tensors
    inputs = {
        'img_1': torch.zeros(size=(4, 3, 224, 224)),
        'img_2': torch.zeros(size=(4, 3, 224, 224)),
    }
    
    # Run forward pass
    outputs = model(inputs)
    
    # Check output structure and shapes
    assert isinstance(outputs, dict), f"Expected dict output, got {type(outputs)}"
    assert 'change_map' in outputs, f"Expected 'change_map' key in outputs, got {outputs.keys()}"
    assert 'aux_change_map' in outputs, f"Expected 'aux_change_map' key in outputs, got {outputs.keys()}"
    assert outputs['change_map'].shape == torch.Size([4, 2, 224, 224]), f"Unexpected shape: {outputs['change_map'].shape}"
    assert outputs['aux_change_map'].shape == torch.Size([4, 2, 224, 224]), f"Unexpected shape: {outputs['aux_change_map'].shape}"

def test_hcgmnet_different_input_size() -> None:
    """Test HCGMNet with different input sizes."""
    model = HCGMNet(num_classes=2)
    model.eval()
    
    # Test with 256x256 input size
    inputs = {
        'img_1': torch.zeros(size=(2, 3, 256, 256)),
        'img_2': torch.zeros(size=(2, 3, 256, 256)),
    }
    
    outputs = model(inputs)
    assert outputs['change_map'].shape == torch.Size([2, 2, 256, 256]), f"Unexpected shape: {outputs['change_map'].shape}"
    
    # Test with non-square input
    inputs = {
        'img_1': torch.zeros(size=(2, 3, 320, 240)),
        'img_2': torch.zeros(size=(2, 3, 320, 240)),
    }
    
    outputs = model(inputs)
    assert outputs['change_map'].shape == torch.Size([2, 2, 320, 240]), f"Unexpected shape: {outputs['change_map'].shape}" 