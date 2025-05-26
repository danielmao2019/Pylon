import torch
import numpy as np
import pytest
from data.transforms.vision_3d.downsample import DownSample


@pytest.fixture
def sample_point_cloud():
    # Create a sample point cloud with multiple fields
    num_points = 1000
    pc = {
        'pos': torch.randn(num_points, 3),  # 3D positions
        'rgb': torch.rand(num_points, 3),   # RGB colors
        'intensity': torch.rand(num_points), # Intensity values
        'feat': torch.randn(num_points, 4)  # Some features
    }
    return pc


def test_downsample_basic(sample_point_cloud):
    # Test basic downsampling with a reasonable voxel size
    voxel_size = 0.1
    downsample = DownSample(voxel_size=voxel_size)
    
    # Apply downsampling
    result = downsample(sample_point_cloud)
    
    # Check that all fields are present
    assert set(result.keys()) == set(sample_point_cloud.keys())
    
    # Check that all tensors have the same number of points
    num_points = result['pos'].shape[0]
    for key, value in result.items():
        assert isinstance(value, torch.Tensor)
        assert value.shape[0] == num_points
    
    # Check that the number of points decreased
    assert num_points <= sample_point_cloud['pos'].shape[0]
    
    # Check that the data types are preserved
    for key in result.keys():
        assert result[key].dtype == sample_point_cloud[key].dtype


def test_downsample_device_consistency(sample_point_cloud):
    # Test that downsampling works on GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Move point cloud to GPU
        gpu_pc = {k: v.to(device) for k, v in sample_point_cloud.items()}
        
        downsample = DownSample(voxel_size=0.1)
        result = downsample(gpu_pc)
        
        # Check that all tensors are on the same device
        for key, value in result.items():
            assert value.device == device


def test_downsample_voxel_size_effect(sample_point_cloud):
    # Test that larger voxel size results in fewer points
    downsample_small = DownSample(voxel_size=0.1)
    downsample_large = DownSample(voxel_size=0.5)
    
    result_small = downsample_small(sample_point_cloud)
    result_large = downsample_large(sample_point_cloud)
    
    # Larger voxel size should result in fewer points
    assert result_large['pos'].shape[0] <= result_small['pos'].shape[0]


def test_downsample_empty_point_cloud():
    # Test with an empty point cloud
    empty_pc = {
        'pos': torch.empty(0, 3),
        'rgb': torch.empty(0, 3),
        'intensity': torch.empty(0)
    }
    
    downsample = DownSample(voxel_size=0.1)
    result = downsample(empty_pc)
    
    # Should return empty tensors
    for key, value in result.items():
        assert value.shape[0] == 0


def test_downsample_single_point():
    # Test with a single point
    single_point_pc = {
        'pos': torch.randn(1, 3),
        'rgb': torch.rand(1, 3),
        'intensity': torch.rand(1)
    }
    
    downsample = DownSample(voxel_size=0.1)
    result = downsample(single_point_pc)
    
    # Should keep the single point
    for key, value in result.items():
        assert value.shape[0] == 1


def test_downsample_invalid_voxel_size():
    # Test with invalid voxel size
    with pytest.raises(ValueError):
        DownSample(voxel_size=0)  # Zero voxel size
    
    with pytest.raises(ValueError):
        DownSample(voxel_size=-1)  # Negative voxel size
