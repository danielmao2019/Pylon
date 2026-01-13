import pytest
import torch

from data.structures.three_d.point_cloud.ops.sampling.grid_sampling_3d_v2 import (
    GridSampling3D,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def test_grid_sampling_3d_basic():
    # Create sample input data
    points = torch.rand(100, 3)  # 100 random 3D points
    change_map = torch.randint(0, 2, (100,))  # Binary labels for each point
    pc = PointCloud(xyz=points, data={'change_map': change_map})

    # Test with mean mode
    sampler_mean = GridSampling3D(size=0.1, mode='mean')
    result_mean = sampler_mean(pc)

    assert isinstance(result_mean, PointCloud)
    assert hasattr(result_mean, 'change_map')
    assert hasattr(result_mean, 'point_indices')
    assert isinstance(result_mean.point_indices, torch.Tensor)
    assert result_mean.point_indices.dim() == 1  # Should be 1D tensor

    # Test with last mode
    sampler_last = GridSampling3D(size=0.1, mode='last')
    result_last = sampler_last(pc)

    assert hasattr(result_last, 'xyz')
    assert hasattr(result_last, 'change_map')
    assert hasattr(result_last, 'point_indices')
    assert isinstance(result_last.point_indices, torch.Tensor)
    assert result_last.point_indices.dim() == 1  # Should be 1D tensor


def test_grid_sampling_3d_edge_cases():
    # Test with single point
    points = torch.rand(1, 3)
    pc = PointCloud(xyz=points)

    sampler = GridSampling3D(size=0.1)
    result = sampler(pc)

    assert isinstance(result.point_indices, torch.Tensor)
    assert result.point_indices.dim() == 1


def test_grid_sampling_3d_invalid_inputs():
    # Test invalid size
    with pytest.raises(ValueError):
        GridSampling3D(size=-1)

    # Test invalid mode
    with pytest.raises(ValueError):
        GridSampling3D(size=0.1, mode='invalid')

    # Test invalid input dimensions
    sampler = GridSampling3D(size=0.1)
    with pytest.raises(AssertionError):
        sampler(None)


def test_grid_sampling_3d_point_indices():
    # Create a simple point cloud with known structure
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],  # Same voxel as first point
            [1.0, 0.0, 0.0],  # Different voxel
            [1.1, 0.0, 0.0],  # Same voxel as third point
        ]
    )
    pc = PointCloud(xyz=points)

    # Test with mean mode
    sampler_mean = GridSampling3D(size=0.5, mode='mean')
    result_mean = sampler_mean(pc)

    assert isinstance(result_mean.point_indices, torch.Tensor)
    assert result_mean.point_indices.dim() == 1
    # Should have indices for all points, grouped by cluster
    assert result_mean.point_indices.shape[0] == pc.num_points

    # Test with last mode
    sampler_last = GridSampling3D(size=0.5, mode='last')
    result_last = sampler_last(pc)

    assert isinstance(result_last.point_indices, torch.Tensor)
    assert result_last.point_indices.dim() == 1
    # Should have indices only for the last point in each cluster
    assert result_last.point_indices.shape[0] == 2  # Two clusters in this case
