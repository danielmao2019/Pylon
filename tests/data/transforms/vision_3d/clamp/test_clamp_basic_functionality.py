import pytest
import torch
from data.transforms.vision_3d.clamp import Clamp


def test_clamp_initialization():
    """Test Clamp transform initialization."""
    # Test valid max_points
    clamp = Clamp(max_points=100)
    assert clamp.max_points == 100
    
    # Test different valid values
    clamp = Clamp(max_points=1)
    assert clamp.max_points == 1
    
    clamp = Clamp(max_points=10000)
    assert clamp.max_points == 10000


def test_clamp_single_large_pc(sample_pc_large):
    """Test Clamp transform with single large point cloud."""
    max_points = 500
    clamp = Clamp(max_points=max_points)
    result = clamp(sample_pc_large, seed=42)
    
    # Check if all original keys are preserved plus 'indices' key added by Select
    expected_keys = set(sample_pc_large.keys()) | {'indices'}
    assert result.keys() == expected_keys
    
    # Check if number of points is correctly clamped
    assert result['pos'].shape[0] == max_points
    assert result['feat'].shape[0] == max_points
    assert result['normal'].shape[0] == max_points
    
    # Check if other dimensions are preserved
    assert result['pos'].shape[1] == 3
    assert result['feat'].shape[1] == 4
    assert result['normal'].shape[1] == 3
    
    # Check device and dtype preservation
    assert result['pos'].device == sample_pc_large['pos'].device
    assert result['pos'].dtype == sample_pc_large['pos'].dtype


def test_clamp_single_small_pc(sample_pc_small):
    """Test Clamp transform with single small point cloud (no clamping needed)."""
    max_points = 100
    clamp = Clamp(max_points=max_points)
    result = clamp(sample_pc_small, seed=42)
    
    # Check if result is identical when no clamping is needed (no 'indices' key added)
    assert result.keys() == sample_pc_small.keys()
    assert torch.equal(result['pos'], sample_pc_small['pos'])
    assert torch.equal(result['feat'], sample_pc_small['feat'])


def test_clamp_deterministic(sample_pc_large):
    """Test if Clamp transform produces consistent results with same seed."""
    max_points = 400
    clamp = Clamp(max_points=max_points)
    
    # Apply transform twice with same seed
    result1 = clamp(sample_pc_large, seed=42)
    result2 = clamp(sample_pc_large, seed=42)
    
    # Results should be identical
    assert torch.equal(result1['pos'], result2['pos'])
    assert torch.equal(result1['feat'], result2['feat'])
    assert torch.equal(result1['normal'], result2['normal'])


def test_clamp_different_seeds(sample_pc_large):
    """Test if Clamp transform produces different results with different seeds."""
    max_points = 400
    clamp = Clamp(max_points=max_points)
    
    # Apply transform with different seeds
    result1 = clamp(sample_pc_large, seed=42)
    result2 = clamp(sample_pc_large, seed=123)
    
    # Results should be different (with very high probability)
    assert not torch.equal(result1['pos'], result2['pos'])


def test_clamp_cpu_device(sample_pc_cpu):
    """Test Clamp transform with CPU tensors."""
    max_points = 200
    clamp = Clamp(max_points=max_points)
    result = clamp(sample_pc_cpu, seed=42)
    
    # Check if device is preserved
    assert result['pos'].device == torch.device('cpu')
    assert result['feat'].device == torch.device('cpu')
    
    # Check if clamping worked correctly
    assert result['pos'].shape[0] == max_points
    assert result['feat'].shape[0] == max_points


def test_clamp_edge_case_exact_max_points(create_pc_factory):
    """Test Clamp transform when point cloud has exactly max_points."""
    max_points = 100
    pc = create_pc_factory(num_points=max_points)
    
    clamp = Clamp(max_points=max_points)
    result = clamp(pc, seed=42)
    
    # Result should be identical to input (no clamping needed)
    assert torch.equal(result['pos'], pc['pos'])
    assert torch.equal(result['feat'], pc['feat'])


def test_clamp_string_representation():
    """Test string representation of Clamp transform."""
    clamp = Clamp(max_points=500)
    str_repr = str(clamp)
    assert "Clamp" in str_repr
    assert "500" in str_repr


@pytest.mark.parametrize("max_points,input_points", [
    (100, 200),    # Clamp to half
    (500, 1000),   # Clamp to half
    (1, 100),      # Clamp to single point
    (999, 1000),   # Clamp to almost all points
])
def test_clamp_various_sizes(max_points, input_points, create_pc_factory):
    """Test Clamp transform with various input and output sizes."""
    pc = create_pc_factory(num_points=input_points)
    clamp = Clamp(max_points=max_points)
    result = clamp(pc, seed=42)
    
    # Check correct clamping
    assert result['pos'].shape[0] == max_points
    assert result['feat'].shape[0] == max_points
    
    # Check dimensions preserved
    assert result['pos'].shape[1] == 3
    assert result['feat'].shape[1] == 4


@pytest.mark.parametrize("device", ['cuda', 'cpu'])
def test_clamp_device_compatibility(device, create_pc_factory):
    """Test Clamp transform with different devices."""
    pc = create_pc_factory(num_points=200, device=device)
    clamp = Clamp(max_points=100)
    result = clamp(pc, seed=42)
    
    # Check device preservation
    assert result['pos'].device.type == device
    assert result['feat'].device.type == device