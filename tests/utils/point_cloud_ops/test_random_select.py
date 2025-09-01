import torch
import pytest
from utils.point_cloud_ops.random_select import RandomSelect


def test_random_select_percentage_basic():
    """Test basic percentage-based selection."""
    pc = {
        'pos': torch.tensor([
            [1.0, 0.0, 0.0],  # Point 0
            [2.0, 0.0, 0.0],  # Point 1
            [3.0, 0.0, 0.0],  # Point 2
            [4.0, 0.0, 0.0],  # Point 3
        ], dtype=torch.float64),
        'rgb': torch.tensor([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
        ], dtype=torch.float64),
    }

    # Select 50% of points
    random_select = RandomSelect(percentage=0.5)
    result = random_select(pc, seed=42)
    
    # Check that approximately 50% of points are selected (2 out of 4)
    expected_count = int(4 * 0.5)  # 2 points
    assert result['pos'].shape[0] == expected_count
    assert result['rgb'].shape[0] == expected_count
    
    # Check that indices key exists and has correct size
    assert 'indices' in result
    assert result['indices'].shape[0] == expected_count
    assert result['indices'].dtype == torch.int64
    
    # Check that all indices are valid
    assert torch.all(result['indices'] >= 0)
    assert torch.all(result['indices'] < 4)


def test_random_select_count_basic():
    """Test basic count-based selection."""
    pc = {
        'pos': torch.tensor([
            [1.0, 0.0, 0.0],  # Point 0
            [2.0, 0.0, 0.0],  # Point 1
            [3.0, 0.0, 0.0],  # Point 2
            [4.0, 0.0, 0.0],  # Point 3
            [5.0, 0.0, 0.0],  # Point 4
        ], dtype=torch.float64),
    }

    # Select exactly 3 points
    random_select = RandomSelect(count=3)
    result = random_select(pc, seed=42)
    
    # Check that exactly 3 points are selected
    assert result['pos'].shape[0] == 3
    
    # Check that indices key exists and has correct size
    assert 'indices' in result
    assert result['indices'].shape[0] == 3
    assert result['indices'].dtype == torch.int64
    
    # Check that all indices are valid and unique
    indices = result['indices']
    assert torch.all(indices >= 0)
    assert torch.all(indices < 5)
    assert len(torch.unique(indices)) == 3  # Should be unique


def test_random_select_deterministic_with_seed():
    """Test that results are deterministic when using same seed."""
    pc = {
        'pos': torch.tensor([
            [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]
        ], dtype=torch.float64),
    }

    random_select = RandomSelect(percentage=0.5)
    
    # Run twice with same seed
    result1 = random_select(pc, seed=42)
    result2 = random_select(pc, seed=42)
    
    # Results should be identical
    assert torch.equal(result1['indices'], result2['indices'])
    assert torch.allclose(result1['pos'], result2['pos'])


def test_random_select_different_with_different_seeds():
    """Test that results are different with different seeds."""
    pc = {
        'pos': torch.tensor([
            [i, 0.0, 0.0] for i in range(20)  # 20 points for higher chance of different results
        ], dtype=torch.float64),
    }

    random_select = RandomSelect(count=10)
    
    # Run with different seeds
    result1 = random_select(pc, seed=42)
    result2 = random_select(pc, seed=123)
    
    # Results should be different (with high probability)
    # Note: There's a small chance they could be the same, but very unlikely
    assert not torch.equal(result1['indices'], result2['indices'])


def test_random_select_generator_parameter():
    """Test using generator parameter instead of seed."""
    pc = {
        'pos': torch.tensor([[i, 0.0, 0.0] for i in range(10)], dtype=torch.float64),
    }

    random_select = RandomSelect(count=5)
    
    # Create generator with specific seed
    generator = torch.Generator()
    generator.manual_seed(42)
    
    result = random_select(pc, generator=generator)
    
    # Should select exactly 5 points
    assert result['pos'].shape[0] == 5
    assert result['indices'].shape[0] == 5


def test_random_select_percentage_edge_cases():
    """Test percentage-based selection edge cases."""
    pc = {
        'pos': torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64),
    }

    # Test 100% selection
    random_select_100 = RandomSelect(percentage=1.0)
    result_100 = random_select_100(pc, seed=42)
    assert result_100['pos'].shape[0] == 2  # All points
    
    # Test very small percentage (uses int() so may select 0 points)
    random_select_small = RandomSelect(percentage=0.1)
    result_small = random_select_small(pc, seed=42)
    expected_small = int(2 * 0.1)  # int(0.2) = 0
    assert result_small['pos'].shape[0] == expected_small


def test_random_select_count_edge_cases():
    """Test count-based selection edge cases."""
    pc = {
        'pos': torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64),
    }

    # Test selecting all points
    random_select_all = RandomSelect(count=3)
    result_all = random_select_all(pc, seed=42)
    assert result_all['pos'].shape[0] == 3
    
    # Test selecting single point
    random_select_one = RandomSelect(count=1)
    result_one = random_select_one(pc, seed=42)
    assert result_one['pos'].shape[0] == 1


def test_random_select_validation_errors():
    """Test input validation errors."""
    # Test that exactly one of percentage or count must be provided
    with pytest.raises(AssertionError, match="Exactly one of percentage or count"):
        RandomSelect()  # Neither provided
    
    with pytest.raises(AssertionError, match="Exactly one of percentage or count"):
        RandomSelect(percentage=0.5, count=10)  # Both provided
    
    # Test invalid percentage values
    with pytest.raises(AssertionError):
        RandomSelect(percentage=0.0)  # Too small
    
    with pytest.raises(AssertionError):
        RandomSelect(percentage=1.5)  # Too large
    
    with pytest.raises(AssertionError):
        RandomSelect(percentage=-0.1)  # Negative
    
    # Test invalid count values
    with pytest.raises(AssertionError):
        RandomSelect(count=0)  # Too small
    
    with pytest.raises(AssertionError):
        RandomSelect(count=-5)  # Negative


def test_random_select_count_exceeds_points():
    """Test behavior when requested count exceeds available points."""
    pc = {
        'pos': torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64),  # Only 2 points
    }

    # Try to select more points than available
    random_select = RandomSelect(count=5)  # More than 2 available
    
    # RandomSelect uses min(count, num_points) so it handles gracefully
    result = random_select(pc, seed=42)
    assert result['pos'].shape[0] == 2  # All available points


def test_random_select_empty_point_cloud():
    """Test that empty point clouds are rejected by input validation."""
    pc = {
        'pos': torch.tensor([], dtype=torch.float64).reshape(0, 3),
        'rgb': torch.tensor([], dtype=torch.float64).reshape(0, 3),
    }

    # Empty point clouds are not allowed by check_point_cloud
    random_select_pct = RandomSelect(percentage=0.5)
    with pytest.raises(AssertionError, match="Expected positive number of points"):
        random_select_pct(pc, seed=42)
    
    random_select_count = RandomSelect(count=1)
    with pytest.raises(AssertionError, match="Expected positive number of points"):
        random_select_count(pc, seed=42)


def test_random_select_uses_select_internally():
    """Test that RandomSelect uses Select internally and inherits index chaining."""
    # Start with PC that already has indices (simulating previous selection)
    pc_with_indices = {
        'pos': torch.tensor([
            [1.0, 0.0, 0.0],  # Original index 0
            [3.0, 0.0, 0.0],  # Original index 2  
            [5.0, 0.0, 0.0],  # Original index 4
        ], dtype=torch.float64),
        'indices': torch.tensor([0, 2, 4], dtype=torch.int64),  # Maps to original PC
    }

    # Select 2 out of 3 points
    random_select = RandomSelect(count=2)
    result = random_select(pc_with_indices, seed=42)
    
    # Should have 2 points
    assert result['pos'].shape[0] == 2
    assert result['indices'].shape[0] == 2
    
    # Final indices should be subset of original indices [0, 2, 4]
    final_indices = result['indices']
    original_indices = pc_with_indices['indices']
    assert torch.all(torch.isin(final_indices, original_indices))


def test_random_select_preserves_all_keys():
    """Test that all point cloud keys are preserved in selection."""
    pc = {
        'pos': torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64),
        'rgb': torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
        'classification': torch.tensor([0, 1, 2], dtype=torch.long),
        'intensity': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
    }

    random_select = RandomSelect(count=2)
    result = random_select(pc, seed=42)
    
    # Check that all keys are preserved
    assert set(result.keys()) == set(pc.keys()) | {'indices'}  # Plus indices key
    
    # Check that tensor dimensions are consistent
    assert result['pos'].shape[0] == 2
    assert result['rgb'].shape[0] == 2
    assert result['classification'].shape[0] == 2
    assert result['intensity'].shape[0] == 2
    
    # Check indices
    assert result['indices'].shape[0] == 2


@pytest.mark.parametrize("percentage", [0.1, 0.25, 0.5, 0.75, 1.0])
def test_random_select_percentage_range(percentage):
    """Test percentage-based selection across range of values."""
    pc = {
        'pos': torch.tensor([[i, 0.0, 0.0] for i in range(100)], dtype=torch.float64),  # 100 points
    }

    random_select = RandomSelect(percentage=percentage)
    result = random_select(pc, seed=42)
    
    expected_count = max(1, int(100 * percentage))  # At least 1, or percentage of 100
    actual_count = result['pos'].shape[0]
    
    # Should be exactly the expected count (ceiling behavior)
    assert actual_count == expected_count
    assert result['indices'].shape[0] == expected_count


@pytest.mark.parametrize("count", [1, 5, 10, 25, 50])
def test_random_select_count_range(count):
    """Test count-based selection across range of values."""
    pc = {
        'pos': torch.tensor([[i, 0.0, 0.0] for i in range(100)], dtype=torch.float64),  # 100 points
    }

    random_select = RandomSelect(count=count)
    result = random_select(pc, seed=42)
    
    # Should be exactly the requested count
    assert result['pos'].shape[0] == count
    assert result['indices'].shape[0] == count
    
    # Indices should be unique
    assert len(torch.unique(result['indices'])) == count