import pytest
import torch


@pytest.fixture
def sample_pc_large():
    """Create a large sample point cloud dictionary for testing."""
    num_points = 1000
    return {
        'pos': torch.randn(size=(num_points, 3), dtype=torch.float32, device='cuda'),
        'feat': torch.randn(size=(num_points, 4), dtype=torch.float32, device='cuda'),
        'normal': torch.randn(size=(num_points, 3), dtype=torch.float32, device='cuda'),
    }


@pytest.fixture
def sample_pc_small():
    """Create a small sample point cloud dictionary for testing."""
    num_points = 50
    return {
        'pos': torch.randn(size=(num_points, 3), dtype=torch.float32, device='cuda'),
        'feat': torch.randn(size=(num_points, 4), dtype=torch.float32, device='cuda'),
    }


@pytest.fixture  
def sample_pc_cpu():
    """Create a CPU sample point cloud dictionary for device testing."""
    num_points = 500
    return {
        'pos': torch.randn(size=(num_points, 3), dtype=torch.float32, device='cpu'),
        'feat': torch.randn(size=(num_points, 4), dtype=torch.float32, device='cpu'),
    }


@pytest.fixture
def create_pc_factory():
    """Factory for creating point clouds with specific parameters."""
    def _create_pc(num_points: int, device: str = 'cuda', include_normal: bool = False):
        pc = {
            'pos': torch.randn(size=(num_points, 3), dtype=torch.float32, device=device),
            'feat': torch.randn(size=(num_points, 4), dtype=torch.float32, device=device),
        }
        if include_normal:
            pc['normal'] = torch.randn(size=(num_points, 3), dtype=torch.float32, device=device)
        return pc
    return _create_pc