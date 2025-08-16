import pytest
import torch
from data.transforms.vision_3d.clamp import Clamp


def test_clamp_multiple_args_consistent_randomness(sample_pc_large):
    """Test Clamp transform with multiple point clouds for consistent randomness."""
    max_points = 300
    clamp = Clamp(max_points=max_points)
    
    # Create two identical point clouds
    pc1 = sample_pc_large.copy()
    pc2 = sample_pc_large.copy()
    
    # Apply clamp to both with same seed
    result1, result2 = clamp(pc1, pc2, seed=42)
    
    # Check if both results have same number of points
    assert result1['pos'].shape[0] == max_points
    assert result2['pos'].shape[0] == max_points
    
    # Check if same indices were selected (consistent randomness)
    # Since both input point clouds are identical, selected points should be identical
    assert torch.equal(result1['pos'], result2['pos'])
    assert torch.equal(result1['feat'], result2['feat'])
    assert torch.equal(result1['normal'], result2['normal'])


def test_clamp_multiple_args_different_data(create_pc_factory):
    """Test Clamp transform with multiple different point clouds."""
    max_points = 200
    clamp = Clamp(max_points=max_points)
    
    # Create two different point clouds with same number of points
    num_points = 500
    pc1 = create_pc_factory(num_points=num_points)
    pc2 = create_pc_factory(num_points=num_points)
    
    # Apply clamp to both with same seed
    result1, result2 = clamp(pc1, pc2, seed=42)
    
    # Check if both results have same number of points
    assert result1['pos'].shape[0] == max_points
    assert result2['pos'].shape[0] == max_points
    
    # Verify that the same indices were selected by checking relative order
    # Generate the expected indices to verify consistency
    generator = torch.Generator(device='cuda')
    generator.manual_seed(42)
    expected_indices = torch.randperm(num_points, generator=generator, device='cuda')[:max_points]
    
    # Check if correct indices were applied
    assert torch.equal(result1['pos'], pc1['pos'][expected_indices])
    assert torch.equal(result1['feat'], pc1['feat'][expected_indices])
    assert torch.equal(result2['pos'], pc2['pos'][expected_indices])
    assert torch.equal(result2['feat'], pc2['feat'][expected_indices])


def test_clamp_no_clamping_multiple_args(sample_pc_small):
    """Test Clamp transform with multiple args when no clamping is needed."""
    max_points = 100  # Larger than sample_pc_small
    clamp = Clamp(max_points=max_points)
    
    pc1 = sample_pc_small.copy()
    pc2 = sample_pc_small.copy()
    
    result1, result2 = clamp(pc1, pc2, seed=42)
    
    # Results should be identical to inputs
    assert torch.equal(result1['pos'], pc1['pos'])
    assert torch.equal(result1['feat'], pc1['feat'])
    assert torch.equal(result2['pos'], pc2['pos'])
    assert torch.equal(result2['feat'], pc2['feat'])


def test_clamp_three_args_consistency(create_pc_factory):
    """Test Clamp transform with three point clouds for consistent randomness."""
    max_points = 150
    clamp = Clamp(max_points=max_points)
    
    # Create three different point clouds with same number of points
    num_points = 400
    pc1 = create_pc_factory(num_points=num_points)
    pc2 = create_pc_factory(num_points=num_points)
    pc3 = create_pc_factory(num_points=num_points)
    
    # Apply clamp to all three with same seed
    result1, result2, result3 = clamp(pc1, pc2, pc3, seed=42)
    
    # Check if all results have same number of points
    assert result1['pos'].shape[0] == max_points
    assert result2['pos'].shape[0] == max_points
    assert result3['pos'].shape[0] == max_points
    
    # Generate expected indices to verify consistency
    generator = torch.Generator(device='cuda')
    generator.manual_seed(42)
    expected_indices = torch.randperm(num_points, generator=generator, device='cuda')[:max_points]
    
    # Check if correct indices were applied to all point clouds
    assert torch.equal(result1['pos'], pc1['pos'][expected_indices])
    assert torch.equal(result1['feat'], pc1['feat'][expected_indices])
    assert torch.equal(result2['pos'], pc2['pos'][expected_indices])
    assert torch.equal(result2['feat'], pc2['feat'][expected_indices])
    assert torch.equal(result3['pos'], pc3['pos'][expected_indices])
    assert torch.equal(result3['feat'], pc3['feat'][expected_indices])


def test_clamp_multi_args_deterministic(create_pc_factory):
    """Test determinism of multi-arg Clamp with same seed."""
    max_points = 200
    clamp = Clamp(max_points=max_points)
    
    # Create two point clouds
    num_points = 500
    pc1 = create_pc_factory(num_points=num_points)
    pc2 = create_pc_factory(num_points=num_points)
    
    # Apply transform twice with same seed
    result1a, result2a = clamp(pc1, pc2, seed=42)
    result1b, result2b = clamp(pc1, pc2, seed=42)
    
    # Results should be identical
    assert torch.equal(result1a['pos'], result1b['pos'])
    assert torch.equal(result1a['feat'], result1b['feat'])
    assert torch.equal(result2a['pos'], result2b['pos'])
    assert torch.equal(result2a['feat'], result2b['feat'])


def test_clamp_multi_args_different_seeds(create_pc_factory):
    """Test multi-arg Clamp produces different results with different seeds."""
    max_points = 200
    clamp = Clamp(max_points=max_points)
    
    # Create two point clouds
    num_points = 500
    pc1 = create_pc_factory(num_points=num_points)
    pc2 = create_pc_factory(num_points=num_points)
    
    # Apply transform with different seeds
    result1a, result2a = clamp(pc1, pc2, seed=42)
    result1b, result2b = clamp(pc1, pc2, seed=123)
    
    # Results should be different (with very high probability)
    assert not torch.equal(result1a['pos'], result1b['pos'])
    assert not torch.equal(result2a['pos'], result2b['pos'])


@pytest.mark.parametrize("num_args", [2, 3, 4, 5])
def test_clamp_multiple_args_scaling(num_args, create_pc_factory):
    """Test Clamp transform with varying numbers of arguments."""
    max_points = 100
    clamp = Clamp(max_points=max_points)
    
    # Create multiple point clouds
    num_points = 300
    point_clouds = [create_pc_factory(num_points=num_points) for _ in range(num_args)]
    
    # Apply clamp to all point clouds
    results = clamp(*point_clouds, seed=42)
    
    # Check if we get the correct number of results
    assert len(results) == num_args
    
    # Check if all results have correct number of points
    for result in results:
        assert result['pos'].shape[0] == max_points
        assert result['feat'].shape[0] == max_points


def test_clamp_multi_args_cpu_device(create_pc_factory):
    """Test multi-arg Clamp transform with CPU tensors."""
    max_points = 100
    clamp = Clamp(max_points=max_points)
    
    # Create two CPU point clouds
    num_points = 250
    pc1 = create_pc_factory(num_points=num_points, device='cpu')
    pc2 = create_pc_factory(num_points=num_points, device='cpu')
    
    result1, result2 = clamp(pc1, pc2, seed=42)
    
    # Check if devices are preserved
    assert result1['pos'].device == torch.device('cpu')
    assert result1['feat'].device == torch.device('cpu')
    assert result2['pos'].device == torch.device('cpu')
    assert result2['feat'].device == torch.device('cpu')
    
    # Check consistent selection
    generator = torch.Generator(device='cpu')
    generator.manual_seed(42)
    expected_indices = torch.randperm(num_points, generator=generator, device='cpu')[:max_points]
    
    assert torch.equal(result1['pos'], pc1['pos'][expected_indices])
    assert torch.equal(result2['pos'], pc2['pos'][expected_indices])