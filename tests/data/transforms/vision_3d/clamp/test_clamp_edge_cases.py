import pytest
import torch
from data.transforms.vision_3d.clamp import Clamp


def test_clamp_empty_point_cloud():
    """Test Clamp transform with empty point cloud (should raise assertion error)."""
    clamp = Clamp(max_points=100)
    
    pc = {
        'pos': torch.empty(0, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.empty(0, 4, dtype=torch.float32, device='cuda'),
    }
    
    # Empty point cloud should raise AssertionError due to validation
    with pytest.raises(AssertionError, match="Expected positive number of points"):
        clamp(pc, seed=42)


def test_clamp_empty_point_cloud_multi_args():
    """Test Clamp transform with multiple empty point clouds (should raise assertion error)."""
    clamp = Clamp(max_points=100)
    
    pc1 = {
        'pos': torch.empty(0, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.empty(0, 4, dtype=torch.float32, device='cuda'),
    }
    pc2 = {
        'pos': torch.empty(0, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.empty(0, 4, dtype=torch.float32, device='cuda'),
    }
    
    # Empty point clouds should raise AssertionError due to validation
    with pytest.raises(AssertionError, match="Expected positive number of points"):
        clamp(pc1, pc2, seed=42)


def test_clamp_single_point():
    """Test Clamp transform with single point (edge case for small point clouds)."""
    clamp = Clamp(max_points=1)
    
    pc = {
        'pos': torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device='cuda'),
        'feat': torch.tensor([[0.5, 0.6, 0.7, 0.8]], dtype=torch.float32, device='cuda'),
    }
    
    # Single point should pass through unchanged (1 <= max_points)
    result = clamp(pc, seed=42)
    assert torch.equal(result['pos'], pc['pos'])
    assert torch.equal(result['feat'], pc['feat'])
    assert result.keys() == pc.keys()  # No indices added when no clamping


def test_clamp_large_max_points(create_pc_factory):
    """Test Clamp transform with max_points much larger than input."""
    # Very large max_points
    clamp = Clamp(max_points=100000)
    
    pc = create_pc_factory(num_points=500)
    result = clamp(pc, seed=42)
    
    # Should pass through unchanged
    assert torch.equal(result['pos'], pc['pos'])
    assert torch.equal(result['feat'], pc['feat'])
    assert result.keys() == pc.keys()  # No indices added when no clamping


def test_clamp_boundary_exact_match(create_pc_factory):
    """Test boundary condition where input points exactly matches max_points."""
    max_points = 250
    clamp = Clamp(max_points=max_points)
    
    pc = create_pc_factory(num_points=max_points)  # Exactly max_points
    result = clamp(pc, seed=42)
    
    # Should pass through unchanged (250 <= 250)
    assert torch.equal(result['pos'], pc['pos'])
    assert torch.equal(result['feat'], pc['feat'])
    assert result.keys() == pc.keys()  # No indices added when no clamping


def test_clamp_boundary_one_over(create_pc_factory):
    """Test boundary condition where input has max_points + 1 points."""
    max_points = 250
    clamp = Clamp(max_points=max_points)
    
    pc = create_pc_factory(num_points=max_points + 1)  # One more than max_points
    result = clamp(pc, seed=42)
    
    # Should be clamped to exactly max_points
    assert result['pos'].shape[0] == max_points
    assert result['feat'].shape[0] == max_points
    # Should have indices added by Select
    expected_keys = set(pc.keys()) | {'indices'}
    assert result.keys() == expected_keys


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clamp_different_dtypes(dtype, create_pc_factory):
    """Test Clamp transform with different float dtypes (both should be accepted)."""
    clamp = Clamp(max_points=100)
    
    pc = {
        'pos': torch.randn(200, 3, dtype=dtype, device='cuda'),
        'feat': torch.randn(200, 4, dtype=torch.float32, device='cuda'),  # feat can be different dtype
    }
    
    result = clamp(pc, seed=42)
    
    # Check that dtype is preserved for pos
    assert result['pos'].dtype == dtype
    assert result['pos'].shape[0] == 100
    assert result['feat'].shape[0] == 100