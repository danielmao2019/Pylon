"""
Tests for the SiameseKPConv model.

This module contains tests for verifying the functionality of the SiameseKPConv model
for 3D point cloud change detection.
"""
import pytest
import torch
import numpy as np
from models.change_detection.siamese_kpconv.siamese_kpconv_model import SiameseKPConv


@pytest.fixture
def test_batch():
    """
    Fixture that creates a test batch with point clouds and change maps.
    
    Returns:
        A dictionary with inputs, labels, and meta_info, structured exactly
        as the model expects for its forward pass.
    """
    torch.manual_seed(42)  # For reproducibility
    
    # Create point clouds - 100 points per cloud, 3 feature dimensions
    batch_size = 2
    num_points = 50
    
    # Create batch
    inputs = {
        'pc_0': {
            'pos': torch.rand(num_points * batch_size, 3),
            'x': torch.rand(num_points * batch_size, 3),
            'batch': torch.cat([
                torch.zeros(num_points, dtype=torch.long),
                torch.ones(num_points, dtype=torch.long)
            ])
        },
        'pc_1': {
            'pos': torch.rand(num_points * batch_size, 3),
            'x': torch.rand(num_points * batch_size, 3),
            'batch': torch.cat([
                torch.zeros(num_points, dtype=torch.long),
                torch.ones(num_points, dtype=torch.long)
            ])
        }
    }
    
    # Create change labels - 1D tensor of shape [num_points * batch_size]
    labels = {
        'change': torch.randint(0, 2, (num_points * batch_size,), dtype=torch.long)
    }
    
    return {
        'inputs': inputs,
        'labels': labels,
        'meta_info': {
            'batch_size': batch_size,
            'num_points': num_points
        }
    }


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


def test_SiameseKPConv_model(model_config, test_batch):
    """Test that the SiameseKPConv model works correctly with test batch."""
    model = SiameseKPConv(**model_config)
    model.eval()
    
    # Forward pass through the model
    with torch.no_grad():
        output = model(test_batch['inputs'])
    
    # Check output shape
    num_points = test_batch['inputs']['pc_1']['pos'].shape[0]
    assert output.shape == (num_points, model_config['out_channels'])
    
    # Check that output contains raw logits (not normalized)
    assert torch.max(output).item() > 1.0 or torch.min(output).item() < 0.0


def test_train_iter(model_config, test_batch):
    """Test a single training iteration with the model."""
    model = SiameseKPConv(**model_config)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Single training step
    model.train()
    
    # Forward pass
    output = model(test_batch['inputs'])
    
    # Calculate loss
    loss = criterion(output, test_batch['labels']['change'])
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check that gradients are computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradients for {name}"
