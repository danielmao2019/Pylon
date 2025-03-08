"""
Tests for the SiameseKPConv model.

This module contains tests for verifying the functionality of the SiameseKPConv model
for 3D point cloud change detection.
"""
import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from models.change_detection.siamese_kpconv.siamese_kpconv_model import SiameseKPConv


@pytest.fixture
def point_cloud_data():
    """Fixture that creates sample point cloud data for testing."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create point cloud 1
    num_points = 100
    pos1 = torch.rand(num_points, 3)  # 3D coordinates
    x1 = torch.rand(num_points, 3)    # Features (e.g., RGB)
    batch1 = torch.zeros(num_points, dtype=torch.long)  # All points in same batch
    
    # Create point cloud 2
    pos2 = torch.rand(num_points, 3)
    x2 = torch.rand(num_points, 3)
    batch2 = torch.zeros(num_points, dtype=torch.long)
    
    # Create Data objects
    data1 = Data(pos=pos1, x=x1, batch=batch1)
    data2 = Data(pos=pos2, x=x2, batch=batch2)
    
    return {'pc_0': data1, 'pc_1': data2, 'num_points': num_points}


@pytest.fixture
def model_config():
    """Fixture that returns a standard model configuration for testing."""
    return {
        'in_channels': 3,
        'out_channels': 2,
        'point_influence': 0.1,
        'down_channels': [16, 32, 64],  # Smaller network for testing
        'up_channels': [64, 32, 16],
        'bn_momentum': 0.02,
        'dropout': 0.1
    }


def test_model_instantiation(model_config):
    """Test that the model can be instantiated with the given configuration."""
    model = SiameseKPConv(**model_config)
    
    # Check model structure
    assert len(model.down_modules) == len(model_config['down_channels'])
    assert len(model.up_modules) == len(model_config['down_channels']) - 1
    assert isinstance(model.inner_modules[0], torch.nn.Identity)


def test_model_forward_pass(point_cloud_data, model_config):
    """Test the forward pass of the model with sample data."""
    model = SiameseKPConv(**model_config)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(point_cloud_data, k=8)
    
    # Check output shape
    num_points = point_cloud_data['num_points']
    expected_shape = (num_points, model_config['out_channels'])
    assert output.shape == expected_shape
    
    # Check that output contains raw logits (not normalized)
    # Raw logits typically have values outside [0,1] range
    assert torch.max(output).item() > 1.0 or torch.min(output).item() < 0.0


def test_model_gradients(point_cloud_data, model_config):
    """Test that gradients flow properly through the model."""
    model = SiameseKPConv(**model_config)
    
    # Forward pass
    output = model(point_cloud_data, k=8)
    
    # Create dummy target
    num_points = point_cloud_data['num_points']
    target = torch.zeros(num_points, dtype=torch.long)
    
    # Compute loss and backpropagate
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    
    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            # Check that gradients are not all zeros or NaNs
            assert not torch.isnan(param.grad).any(), f"NaN gradients for {name}"
            assert not (param.grad == 0).all(), f"Zero gradients for {name}"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(model_config, batch_size):
    """Test the model with different batch sizes using parametrize."""
    model = SiameseKPConv(**model_config)
    model.eval()
    
    # Create data with given batch size
    num_points = 50 * batch_size
    pos1 = torch.rand(num_points, 3)
    x1 = torch.rand(num_points, 3)
    batch1 = torch.repeat_interleave(torch.arange(batch_size), 50)
    
    pos2 = torch.rand(num_points, 3)
    x2 = torch.rand(num_points, 3)
    batch2 = torch.repeat_interleave(torch.arange(batch_size), 50)
    
    # Create Data objects
    data1 = Data(pos=pos1, x=x1, batch=batch1)
    data2 = Data(pos=pos2, x=x2, batch=batch2)
    
    # Forward pass
    with torch.no_grad():
        output = model({'pc_0': data1, 'pc_1': data2}, k=8)
    
    # Check output shape
    expected_shape = (num_points, model_config['out_channels'])
    assert output.shape == expected_shape


@pytest.mark.parametrize("in_channels", [1, 3, 6])
def test_different_input_dimensions(point_cloud_data, in_channels):
    """Test the model with different input feature dimensions using parametrize."""
    # Create data with different feature dimensions
    num_points = point_cloud_data['num_points']
    data1 = point_cloud_data['pc_0']
    data2 = point_cloud_data['pc_1']
    
    # Override feature dimensions
    data1.x = torch.rand(num_points, in_channels)
    data2.x = torch.rand(num_points, in_channels)
    
    # Create model
    model = SiameseKPConv(
        in_channels=in_channels,
        out_channels=2,
        point_influence=0.1,
        down_channels=[16, 32],  # Small network for testing
        up_channels=[32, 16]
    )
    
    # Forward pass
    with torch.no_grad():
        output = model({'pc_0': data1, 'pc_1': data2}, k=8)
    
    # Check output shape
    expected_shape = (num_points, 2)  # 2 output classes
    assert output.shape == expected_shape
