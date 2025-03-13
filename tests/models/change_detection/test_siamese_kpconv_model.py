"""
Tests for the SiameseKPConv model.

This module contains tests for verifying the functionality of the SiameseKPConv model
for 3D point cloud change detection, with specific focus on feature dimensions
through the network.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from models.change_detection.siamese_kpconv.siamese_kpconv_model import SiameseKPConv
from models.change_detection.siamese_kpconv.utils import knn


@pytest.fixture
def test_batch():
    """
    Fixture that creates a minimal test batch with point clouds.
    Specifically structured to test feature dimension handling.
    """
    torch.manual_seed(42)  # For reproducibility
    
    # Create point clouds with minimal points but correct feature structure
    batch_size = 1
    num_points = 100  # Small but enough to test convolutions
    
    # Create batch with ones feature as required by the model
    inputs = {
        'pc_0': {
            'pos': torch.rand(num_points * batch_size, 3),
            'feat': torch.ones(num_points * batch_size, 1),  # ones feature
            'batch': torch.zeros(num_points * batch_size, dtype=torch.long)
        },
        'pc_1': {
            'pos': torch.rand(num_points * batch_size, 3),
            'feat': torch.ones(num_points * batch_size, 1),  # ones feature
            'batch': torch.zeros(num_points * batch_size, dtype=torch.long)
        }
    }
    
    return inputs


def test_feature_dimensions(test_batch):
    """
    Test specifically focused on feature dimensions through the network.
    This test verifies that the model handles feature dimensions correctly
    throughout the network, especially with skip connections.
    """
    # Create model with the full configuration
    model = SiameseKPConv(
        in_channels=4,  # 3 for XYZ + 1 for ones feature
        out_channels=7,
        point_influence=0.05,
        down_channels=[64, 128, 256, 512, 1024],
        up_channels=[1024, 512, 256, 128, 64],
        bn_momentum=0.02,
        dropout=0.5,
        conv_type='simple',
        block_params={
            'n_kernel_points': 25,
            'max_num_neighbors': 25,
        }
    )
    
    # Forward pass through the model
    output = model(test_batch)
    
    # Check output dimensions
    num_points = test_batch['pc_0']['pos'].size(0)
    assert output.shape == (num_points, 7), \
        f"Expected output shape ({num_points}, 7), got {output.shape}"
    
    # Check that output contains valid values
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"
    
    # Check that the model preserved the last feature
    assert model.last_feature is not None, "Model did not preserve last feature"
    assert model.last_feature.shape == (num_points, 64), \
        f"Expected last feature shape ({num_points}, 64), got {model.last_feature.shape}"


def test_feature_dimensions_step_by_step(test_batch):
    """
    Test that steps through the network examining feature dimensions at each stage.
    """
    model = SiameseKPConv(
        in_channels=4,
        out_channels=7,
        point_influence=0.05,
        down_channels=[64, 128],  # Simplified for testing
        up_channels=[128, 64],
        bn_momentum=0.02,
        dropout=0.5,
        conv_type='simple'
    )
    
    # Get the initial features
    x1 = torch.cat([test_batch['pc_0']['pos'], test_batch['pc_0']['feat']], dim=1)
    x2 = torch.cat([test_batch['pc_1']['pos'], test_batch['pc_1']['feat']], dim=1)
    
    # Check initial dimensions
    assert x1.shape[1] == 4, f"Expected 4 input channels, got {x1.shape[1]}"
    assert x2.shape[1] == 4, f"Expected 4 input channels, got {x2.shape[1]}"
    
    # Process through first down module
    pos1, batch1 = test_batch['pc_0']['pos'], test_batch['pc_0']['batch']
    pos2, batch2 = test_batch['pc_1']['pos'], test_batch['pc_1']['batch']
    
    x1_down = model.down_modules[0](x1, pos1, batch1, pos1, batch1, k=16)
    x2_down = model.down_modules[0](x2, pos2, batch2, pos2, batch2, k=16)
    
    # Check dimensions after first down module
    assert x1_down.shape[1] == 64, f"Expected 64 channels after first down module, got {x1_down.shape[1]}"
    assert x2_down.shape[1] == 64, f"Expected 64 channels after first down module, got {x2_down.shape[1]}"
    
    # Process through second down module
    x1_down2 = model.down_modules[1](x1_down, pos1, batch1, pos1, batch1, k=16)
    x2_down2 = model.down_modules[1](x2_down, pos2, batch2, pos2, batch2, k=16)
    
    # Check dimensions after second down module
    assert x1_down2.shape[1] == 128, f"Expected 128 channels after second down module, got {x1_down2.shape[1]}"
    assert x2_down2.shape[1] == 128, f"Expected 128 channels after second down module, got {x2_down2.shape[1]}"
    
    # Calculate difference features for skip connection
    row_idx, col_idx = knn(pos2, pos1, 1, batch2, batch1)
    diff = x2_down2 - x1_down2[col_idx]
    
    # Check dimensions after difference
    assert diff.shape[1] == 128, f"Expected 128 channels after difference, got {diff.shape[1]}"
    
    # Get corresponding skip connection from encoder
    skip_x = x2_down - x1_down[col_idx]  # First layer skip connection
    
    # Concatenate skip connection features
    x = torch.cat([diff, skip_x], dim=1)
    
    # Check dimensions after concatenation
    assert x.shape[1] == 192, f"Expected 192 channels after concatenation (128 + 64), got {x.shape[1]}"
    
    # Process through up module with skip connection
    x = model.up_modules[0](x, pos2, batch2, pos2, batch2, k=16)
    
    # Check dimensions after up module
    assert x.shape[1] == 64, f"Expected 64 channels after up module, got {x.shape[1]}" 