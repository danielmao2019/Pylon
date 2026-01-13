import pytest
import torch
from data.structures.three_d.point_cloud.point_cloud import PointCloud


@pytest.fixture
def sample_pc_large():
    """Create a large sample point cloud dictionary for testing."""
    num_points = 1000
    data = {
        'xyz': torch.randn(size=(num_points, 3), dtype=torch.float32, device='cuda'),
        'feat': torch.randn(size=(num_points, 4), dtype=torch.float32, device='cuda'),
        'normal': torch.randn(size=(num_points, 3), dtype=torch.float32, device='cuda'),
    }
    return PointCloud(data=data)


@pytest.fixture
def sample_pc_small():
    """Create a small sample point cloud dictionary for testing."""
    num_points = 50
    data = {
        'xyz': torch.randn(size=(num_points, 3), dtype=torch.float32, device='cuda'),
        'feat': torch.randn(size=(num_points, 4), dtype=torch.float32, device='cuda'),
    }
    return PointCloud(data=data)


@pytest.fixture
def sample_pc_cpu():
    """Create a CPU sample point cloud dictionary for device testing."""
    num_points = 500
    data = {
        'xyz': torch.randn(size=(num_points, 3), dtype=torch.float32, device='cpu'),
        'feat': torch.randn(size=(num_points, 4), dtype=torch.float32, device='cpu'),
    }
    return PointCloud(data=data)


@pytest.fixture
def create_pc_factory():
    """Factory for creating point clouds with specific parameters."""
    def _create_pc(num_points: int, device: str = 'cuda', include_normal: bool = False):
        data = {
            'xyz': torch.randn(size=(num_points, 3), dtype=torch.float32, device=device),
            'feat': torch.randn(size=(num_points, 4), dtype=torch.float32, device=device),
        }
        if include_normal:
            data['normal'] = torch.randn(size=(num_points, 3), dtype=torch.float32, device=device)
        return PointCloud(data=data)
    return _create_pc
