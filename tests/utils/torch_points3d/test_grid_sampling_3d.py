import torch
import pytest
from utils.torch_points3d.grid_sampling_3d import GridSampling3D


def test_grid_sampling_3d_basic():
    # Create sample input data
    points = torch.rand(100, 3)  # 100 random 3D points
    data_dict = {
        'pos': points,
        'change_map': torch.randint(0, 2, (100,))  # Binary labels for each point
    }
    
    # Test with mean mode
    sampler_mean = GridSampling3D(size=0.1, mode='mean')
    result_mean = sampler_mean(data_dict)
    
    assert 'pos' in result_mean
    assert 'change_map' in result_mean
    assert 'point_indices' in result_mean
    assert isinstance(result_mean['point_indices'], torch.Tensor)
    assert result_mean['point_indices'].dim() == 1  # Should be 1D tensor
    
    # Test with last mode
    sampler_last = GridSampling3D(size=0.1, mode='last')
    result_last = sampler_last(data_dict)
    
    assert 'pos' in result_last
    assert 'change_map' in result_last
    assert 'point_indices' in result_last
    assert isinstance(result_last['point_indices'], torch.Tensor)
    assert result_last['point_indices'].dim() == 1  # Should be 1D tensor


def test_grid_sampling_3d_edge_cases():
    # Test with single point
    points = torch.rand(1, 3)
    data_dict = {'pos': points}
    
    sampler = GridSampling3D(size=0.1)
    result = sampler(data_dict)
    
    assert isinstance(result['point_indices'], torch.Tensor)
    assert result['point_indices'].dim() == 1
    
    # Test with empty point cloud
    points = torch.zeros((0, 3))
    data_dict = {'pos': points}
    
    sampler = GridSampling3D(size=0.1)
    with pytest.raises(IndexError):
        result = sampler(data_dict)


def test_grid_sampling_3d_invalid_inputs():
    # Test invalid size
    with pytest.raises(ValueError):
        GridSampling3D(size=-1)
    
    # Test invalid mode
    with pytest.raises(ValueError):
        GridSampling3D(size=0.1, mode='invalid')
    
    # Test invalid input dimensions
    points = torch.rand(100, 2)  # Only 2D points
    data_dict = {'pos': points}
    
    sampler = GridSampling3D(size=0.1)
    with pytest.raises(ValueError):
        result = sampler(data_dict)


def test_grid_sampling_3d_point_indices():
    # Create a simple point cloud with known structure
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],  # Same voxel as first point
        [1.0, 0.0, 0.0],  # Different voxel
        [1.1, 0.0, 0.0],  # Same voxel as third point
    ])
    data_dict = {'pos': points}

    # Test with mean mode
    sampler_mean = GridSampling3D(size=0.5, mode='mean')
    result_mean = sampler_mean(data_dict)
    
    assert isinstance(result_mean['point_indices'], torch.Tensor)
    assert result_mean['point_indices'].dim() == 1
    # Should have indices for all points, grouped by cluster
    assert result_mean['point_indices'].shape[0] == points.shape[0]
    
    # Test with last mode
    sampler_last = GridSampling3D(size=0.5, mode='last')
    result_last = sampler_last(data_dict)
    
    assert isinstance(result_last['point_indices'], torch.Tensor)
    assert result_last['point_indices'].dim() == 1
    # Should have indices only for the last point in each cluster
    assert result_last['point_indices'].shape[0] == 2  # Two clusters in this case
