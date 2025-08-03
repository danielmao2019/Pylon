"""
Unit tests for PARENet model integration.

Tests PARENet model instantiation, forward pass, and integration with Pylon framework.
"""

import pytest
import torch
from utils.builders import build_from_config
from configs.common.models.point_cloud_registration.parenet_cfg import model_cfg


def test_parenet_model_instantiation():
    """Test that PARENet model can be instantiated from config."""
    model = build_from_config(model_cfg)
    assert model is not None
    assert hasattr(model, 'parenet_model')
    assert hasattr(model, 'cfg')


def test_parenet_model_forward_pass():
    """Test PARENet model forward pass with properly formatted data."""
    model = build_from_config(model_cfg)
    model.eval()
    
    # Create properly formatted input matching PARENet collator output
    batch_size = 2
    total_points = 200  # 100 points per pair
    
    # PARENet expects collated data with specific structure
    dummy_input = {
        'points': torch.randn(total_points, 3),  # Concatenated points
        'lengths': torch.tensor([100, 100]),     # Points per cloud
        'features': torch.randn(total_points, 1), # Concatenated features
        'batch_size': batch_size
    }
    
    with torch.no_grad():
        output = model(dummy_input)
        
        # Verify output structure
        assert isinstance(output, dict), "Model output must be a dictionary"
        
        # Check expected output keys based on PARENet model documentation
        expected_keys = ['estimated_transform', 'ref_corr_points', 'src_corr_points', 
                        'coarse_precision', 'fine_precision', 'rmse', 'registration_recall']
        
        for key in expected_keys:
            assert key in output, f"Missing expected output key: {key}"
            assert isinstance(output[key], torch.Tensor), f"Output {key} must be a tensor"
        
        # Check tensor shapes
        assert output['estimated_transform'].shape == (4, 4), "Transform must be 4x4 matrix"
        assert output['coarse_precision'].dim() == 0, "Coarse precision must be scalar"
        assert output['fine_precision'].dim() == 0, "Fine precision must be scalar"


def test_parenet_model_device_handling():
    """Test PARENet model device handling."""
    model = build_from_config(model_cfg)
    
    # Test CPU mode
    assert next(model.parameters()).device.type == 'cpu'
    
    # Test CUDA mode (CUDA must be available)
    model_cuda = model.cuda()
    assert next(model_cuda.parameters()).device.type == 'cuda'


def test_parenet_model_parameter_count():
    """Test PARENet model has reasonable parameter count."""
    model = build_from_config(model_cfg)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # PARENet should have a significant number of parameters (> 1M)
    assert total_params > 1_000_000, f"Model has too few parameters: {total_params}"
    assert trainable_params > 1_000_000, f"Model has too few trainable parameters: {trainable_params}"
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def test_parenet_model_backbone_structure():
    """Test PARENet model backbone structure."""
    model = build_from_config(model_cfg)
    
    # Check that the model has the expected backbone structure
    assert hasattr(model.parenet_model, 'backbone')
    backbone = model.parenet_model.backbone
    
    # Check backbone has expected components
    expected_backbone_attrs = ['encoder2_1', 'encoder2_2', 'encoder2_3']
    for attr in expected_backbone_attrs:
        assert hasattr(backbone, attr), f"Backbone missing {attr}"
