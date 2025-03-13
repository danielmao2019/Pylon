"""
Tests for the Siam3DCDNet model.

This module contains tests for verifying the functionality of the Siam3DCDNet model
for 3D point cloud change detection based on the 3DCDNet paper.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any

from models.change_detection.siam3dcdnet.siam3dcdnet_model import Siam3DCDNet


@pytest.fixture
def test_batch():
    """
    Fixture that creates a minimal test batch with point clouds.
    Specifically structured to test feature dimension handling.
    """
    torch.manual_seed(42)  # For reproducibility
    
    # Set parameters
    batch_size = 2
    num_points = 1024
    num_features = 3  # XYZ coordinates
    
    # Define downsampling ratios
    sub_sampling_ratio = [4, 4, 4, 4]
    
    # Calculate point counts at each level
    point_counts = [num_points]
    for ratio in sub_sampling_ratio:
        point_counts.append(point_counts[-1] // ratio)
    
    # Number of nearest neighbors
    k_neighbors = 16
    
    # Create xyz list (point coordinates at each level)
    xyz_list = []
    for level_points in point_counts:
        # Create random point coordinates
        xyz = torch.rand(batch_size, level_points, num_features)
        xyz_list.append(xyz)
    
    # Create neighbor indices for each level
    neighbor_idx_list = []
    for level_points in point_counts:
        # Random indices for k nearest neighbors at each point
        neighbors = torch.randint(0, level_points, (batch_size, level_points, k_neighbors))
        neighbor_idx_list.append(neighbors)
    
    # Create pooling indices between levels
    pool_idx_list = []
    for i in range(len(point_counts) - 1):
        # Indices for pooling from level i to level i+1
        # Each point in level i+1 has k_neighbors corresponding points in level i
        pool_idx = torch.randint(0, point_counts[i], 
                                  (batch_size, point_counts[i+1], k_neighbors))
        pool_idx_list.append(pool_idx)
    
    # Create upsampling indices between levels
    unsam_idx_list = []
    for i in range(len(point_counts) - 1):
        # Indices for upsampling from level i+1 to level i
        # Each point in level i has a corresponding point in level i+1
        unsam_idx = torch.randint(0, point_counts[i+1], 
                                   (batch_size, point_counts[i], 1))
        unsam_idx_list.append(unsam_idx)
    
    # Create cross-cloud KNN indices
    # These are indices of k nearest points in the other point cloud
    knearest_idx_0_to_1 = torch.randint(0, num_points, (batch_size, num_points, k_neighbors))
    knearest_idx_1_to_0 = torch.randint(0, num_points, (batch_size, num_points, k_neighbors))
    
    # Assemble the data dict for point cloud 0
    pc_0 = {
        'xyz': xyz_list,
        'neighbors_idx': neighbor_idx_list,
        'pool_idx': pool_idx_list,
        'unsam_idx': unsam_idx_list,
        'knearst_idx_in_another_pc': knearest_idx_0_to_1
    }
    
    # Assemble the data dict for point cloud 1
    pc_1 = {
        'xyz': xyz_list.copy(),  # Creating a copy to simulate a different point cloud
        'neighbors_idx': neighbor_idx_list.copy(),
        'pool_idx': pool_idx_list.copy(),
        'unsam_idx': unsam_idx_list.copy(),
        'knearst_idx_in_another_pc': knearest_idx_1_to_0
    }
    
    # Final data dict
    data_dict = {
        'pc_0': pc_0,
        'pc_1': pc_1
    }
    
    return data_dict


def test_model_instantiation():
    """Test that the model can be instantiated with default parameters."""
    model = Siam3DCDNet(num_classes=2, input_dim=3)
    assert isinstance(model, Siam3DCDNet)
    assert model.num_classes == 2
    assert model.input_dim == 3
    assert model.k_neighbors == 16
    assert model.sub_sampling_ratio == [4, 4, 4, 4]


def test_model_forward_pass(test_batch):
    """Test the forward pass of the model with a test batch."""
    model = Siam3DCDNet(num_classes=2, input_dim=3)
    with torch.no_grad():
        outputs = model(test_batch)
    
    # Check that output is a dictionary
    assert isinstance(outputs, dict)
    
    # Check dictionary keys
    expected_keys = ['logits_0', 'logits_1']
    assert set(outputs.keys()) == set(expected_keys)
    
    # Check output tensors - type, shape, dtype
    batch_size = 2
    num_points = 1024
    num_classes = 2
    
    for key in outputs:
        # Check tensor type
        assert isinstance(outputs[key], torch.Tensor)
        
        # Check shape
        expected_shape = (batch_size, num_points, num_classes)
        assert outputs[key].shape == expected_shape
        
        # Check dtype
        assert outputs[key].dtype == torch.float32


def test_nearest_feature_difference():
    """Test the nearest_feature_difference method of the model."""
    batch_size = 2
    num_points = 100
    feature_dim = 64
    k_neighbors = 16
    
    # Create dummy feature tensors
    raw = torch.rand(batch_size, feature_dim, num_points, 1)
    query = torch.rand(batch_size, feature_dim, num_points, 1)
    
    # Create dummy nearest indices
    nearest_idx = torch.randint(0, num_points, (batch_size, num_points, k_neighbors))
    
    # Call the method
    result = Siam3DCDNet.nearest_feature_difference(raw, query, nearest_idx)
    
    # Check the result shape
    assert result.shape == (batch_size, feature_dim, num_points)
    
    # Check the result dtype
    assert result.dtype == torch.float32


def test_c3dnet_feature_extraction():
    """Test the feature extraction functionality of the C3Dnet backbone."""
    batch_size = 2
    num_points = 1024  # Larger number to prevent zeros after downsampling
    num_features = 3
    k_neighbors = 16
    
    # Create a simplified test input for the C3Dnet backbone
    xyz_list = [torch.rand(batch_size, num_points, num_features)]
    for _ in range(4):  # Add 4 more levels with decreasing point counts
        num_points = num_points // 4
        xyz_list.append(torch.rand(batch_size, num_points, num_features))
    
    # Create neighbor indices
    neighbor_idx_list = []
    for i in range(5):  # 5 levels
        level_points = xyz_list[i].shape[1]
        # Ensure we have at least 1 point to select from
        neighbors = torch.randint(0, max(1, level_points), 
                               (batch_size, level_points, k_neighbors))
        neighbor_idx_list.append(neighbors)
    
    # Create pooling indices
    pool_idx_list = []
    for i in range(4):  # 4 transitions between levels
        # Ensure we have at least 1 point to select from
        pool_idx = torch.randint(0, max(1, xyz_list[i].shape[1]), 
                              (batch_size, xyz_list[i+1].shape[1], k_neighbors))
        pool_idx_list.append(pool_idx)
    
    # Create upsampling indices
    unsam_idx_list = []
    for i in range(4):  # 4 transitions between levels
        # Ensure we have at least 1 point to select from
        unsam_idx = torch.randint(0, max(1, xyz_list[i+1].shape[1]), 
                               (batch_size, xyz_list[i].shape[1], 1))
        unsam_idx_list.append(unsam_idx)
    
    # Assemble the input for C3Dnet
    end_points = [xyz_list, neighbor_idx_list, pool_idx_list, unsam_idx_list]
    
    # Create and test the C3Dnet
    from models.change_detection.siam3dcdnet.siam3dcdnet_model import C3Dnet
    model = C3Dnet(in_d=3, out_d=64)
    
    with torch.no_grad():
        output = model(end_points)
    
    # Check the output shape
    assert output.shape == (batch_size, 64, xyz_list[0].shape[1], 1)
    assert output.dtype == torch.float32
