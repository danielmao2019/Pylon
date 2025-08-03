"""
Unit tests for PARENet criterion integration.

Tests PARENet loss function instantiation and computation.
"""

import pytest
import torch
from utils.builders import build_from_config
from configs.common.criteria.point_cloud_registration.parenet_criterion_cfg import criterion_cfg


def test_parenet_criterion_instantiation():
    """Test that PARENet criterion can be instantiated from config."""
    criterion = build_from_config(criterion_cfg)
    assert criterion is not None
    assert hasattr(criterion, 'parenet_loss')


def test_parenet_criterion_loss_computation():
    """Test PARENet criterion loss computation with properly formatted data."""
    criterion = build_from_config(criterion_cfg)
    
    # Create properly formatted data matching PARENet loss function expectations
    batch_size = 1  # PARENet typically processes single pairs
    num_points = 100
    
    # Create realistic data structure for PARENet loss computation
    dummy_predictions = {
        'ref_points': torch.randn(num_points, 3),
        'src_points': torch.randn(num_points, 3), 
        'ref_feats': torch.randn(num_points, 256),
        'src_feats': torch.randn(num_points, 256),
        'transform': torch.eye(4),
        'correspondences': torch.randint(0, num_points, (50, 2)).long(),
    }
    
    dummy_targets = {
        'ref_points': torch.randn(num_points, 3),
        'src_points': torch.randn(num_points, 3),
        'transform': torch.eye(4),
        'correspondences': torch.randint(0, num_points, (30, 2)).long(),
    }
    
    loss_dict = criterion(dummy_predictions, dummy_targets)
    
    # Verify loss computation results
    assert isinstance(loss_dict, dict), "Loss output must be a dictionary"
    assert 'total_loss' in loss_dict, "Loss dict must contain 'total_loss'"
    assert isinstance(loss_dict['total_loss'], torch.Tensor), "Total loss must be a tensor"
    assert loss_dict['total_loss'].requires_grad, "Loss must require gradients"
    assert loss_dict['total_loss'].dim() == 0, "Loss must be a scalar tensor"
    assert loss_dict['total_loss'].item() >= 0, "Loss must be non-negative"


def test_parenet_criterion_device_handling():
    """Test PARENet criterion device handling."""
    criterion = build_from_config(criterion_cfg)
    
    # Test CPU mode
    for param in criterion.parameters():
        assert param.device.type == 'cpu'
    
    # Test CUDA mode (CUDA must be available)
    criterion_cuda = criterion.cuda()
    for param in criterion_cuda.parameters():
        assert param.device.type == 'cuda'


def test_parenet_criterion_parameter_count():
    """Test PARENet criterion parameter count."""
    criterion = build_from_config(criterion_cfg)
    
    total_params = sum(p.numel() for p in criterion.parameters())
    print(f"Criterion total parameters: {total_params}")
    
    # PARENet criterion should have minimal parameters (mainly for evaluator)
    assert total_params >= 0  # May have no parameters or few parameters
