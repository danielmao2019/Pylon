"""
Tests for the SiameseKPConv model.

This module contains tests for verifying the functionality of the SiameseKPConv model
for 3D point cloud change detection.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from models.change_detection.siamese_kpconv.siamese_kpconv_model import SiameseKPConv


@pytest.fixture
def test_batch():
    """
    Fixture that creates a minimal test batch with point clouds.
    """
    torch.manual_seed(42)  # For reproducibility
    
    # Create point clouds - small number of points for testing
    batch_size = 2
    num_points = 10  # Very small for faster testing
    
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
        'labels': labels
    }


def test_SiameseKPConv_construction():
    """Test that we can construct the model without errors."""
    model = SiameseKPConv(
        in_channels=3,
        out_channels=2,
        down_channels=[16],
        up_channels=[16],
        conv_type='simple'
    )
    
    # Check model has the expected modules
    assert hasattr(model, 'down_modules')
    assert hasattr(model, 'up_modules')
    assert hasattr(model, 'FC_layer')


def test_minimal_forward_pass(test_batch):
    """Test a minimal forward pass with our SiameseKPConv model."""
    # Create a minimal model configuration
    model = SiameseKPConv(
        in_channels=3,
        out_channels=2,
        down_channels=[8],  # Very minimal for testing
        up_channels=[8],  # Very minimal for testing
        point_influence=0.1,
        bn_momentum=0.1,
        dropout=0.0,  # No dropout for deterministic testing
        conv_type='simple'
    )
    model.eval()
    
    # Forward pass through the model
    with torch.no_grad():
        output = model(test_batch['inputs'])
    
    # Check output shape
    num_points = test_batch['inputs']['pc_1']['x'].shape[0]
    assert output.shape == (num_points, 2)  # out_channels = 2


@pytest.fixture
def model_config():
    """Fixture that returns a minimal model configuration for testing."""
    return {
        'in_channels': 3,
        'out_channels': 2,
        'point_influence': 0.1,
        # Use a single layer network for simplicity
        'down_channels': [8],  # Very minimal for testing
        'up_channels': [8],  # Very minimal for testing
        'bn_momentum': 0.1,
        'dropout': 0.0,  # No dropout for deterministic testing
        'conv_type': 'simple'
    }


def test_SiameseKPConv_model(model_config, test_batch):
    """Test that the SiameseKPConv model works correctly with test batch."""
    # Use the same minimal configuration as in test_minimal_forward_pass
    model = SiameseKPConv(**model_config)
    model.eval()
    
    # Forward pass through the model
    with torch.no_grad():
        output = model(test_batch['inputs'])
    
    # Check output shape
    num_points = test_batch['inputs']['pc_1']['x'].shape[0]
    assert output.shape == (num_points, model_config['out_channels'])
    
    # Check that output contains raw logits (not normalized)
    # Raw logits can have any range of values
    assert output.requires_grad == False  # Should be detached in eval mode


def test_train_iter(model_config, test_batch):
    """Test a single training iteration with the model."""
    # Use the same minimal configuration as in test_minimal_forward_pass
    model = SiameseKPConv(**model_config)
    
    # Set up optimizer with small learning rate for stability
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Use CrossEntropyLoss since our model returns raw logits
    criterion = torch.nn.CrossEntropyLoss()
    
    # Single training step
    model.train()
    
    # Forward pass
    output = model(test_batch['inputs'])
    
    # Calculate loss - making sure the labels are right shape
    loss = criterion(output, test_batch['labels']['change'])
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check that at least some gradients are computed (we can't ensure all parameters get gradients)
    has_grads = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if not torch.isnan(param.grad).any():
                has_grads = True
                break
    
    assert has_grads, "No valid gradients found in any parameter"
