import pytest
import torch
from data.transforms.vision_3d.scale import Scale


@pytest.fixture
def sample_pc():
    """Create a sample point cloud dictionary for testing."""
    num_points = 1000
    return {
        'pos': torch.randn(size=(num_points, 3), device='cuda'),
        'feat': torch.randn(size=(num_points, 4), device='cuda'),
        'normal': torch.randn(size=(num_points, 3), device='cuda'),
    }


def test_scale_init():
    """Test Scale transform initialization."""
    # Test valid scale factor
    scale = Scale(scale_factor=0.1)
    assert scale.scale_factor == 0.1

    # Test invalid scale factor types
    with pytest.raises(AssertionError):
        Scale(scale_factor="invalid")

    # Test invalid scale factor values
    with pytest.raises(AssertionError):
        Scale(scale_factor=0.0)
    with pytest.raises(AssertionError):
        Scale(scale_factor=1.0)
    with pytest.raises(AssertionError):
        Scale(scale_factor=2.0)


def test_scale_call(sample_pc):
    """Test Scale transform call with valid input."""    
    # Use a larger scale factor (0.5) that will result in 125 points (0.5^3 * 1000)
    scale = Scale(scale_factor=0.5)
    result = scale(sample_pc, seed=42)

    # Check if all keys are preserved
    assert result.keys() == sample_pc.keys()

    # Check if shapes are consistent
    expected_num_points = int(sample_pc['pos'].shape[0] * (0.5 ** 3))
    assert result['pos'].shape == (expected_num_points, 3)
    assert result['feat'].shape == (expected_num_points, 4)
    assert result['normal'].shape == (expected_num_points, 3)

    # Get expected indices for deterministic comparison
    torch.manual_seed(42)
    indices = torch.randperm(sample_pc['pos'].shape[0], device=sample_pc['pos'].device)[:expected_num_points]

    # Check if XYZ coordinates are scaled
    assert torch.allclose(result['pos'], sample_pc['pos'][indices] * 0.5)
    assert torch.allclose(result['feat'], sample_pc['feat'][indices])
    assert torch.allclose(result['normal'], sample_pc['normal'][indices])


def test_scale_invalid_input():
    """Test Scale transform with invalid inputs."""
    scale = Scale(scale_factor=0.1)

    # Test with non-dictionary input
    with pytest.raises(AssertionError):
        scale("not a dict")

    # Test with missing 'pos' key
    with pytest.raises(AssertionError):
        scale({'feat': torch.randn(10, 4)})

    # Test with non-tensor values
    with pytest.raises(AssertionError):
        scale({
            'pos': torch.randn(10, 3),
            'feat': "not a tensor"
        })

    # Test with inconsistent shapes
    with pytest.raises(AssertionError):
        scale({
            'pos': torch.randn(10, 3),
            'feat': torch.randn(8, 4)  # Different number of points
        })


def test_scale_deterministic():
    """Test if Scale transform produces consistent results with same input."""
    scale = Scale(scale_factor=0.1)
    pc = {
        'pos': torch.randn(size=(1000, 3), device='cuda'),
        'feat': torch.randn(size=(1000, 4), device='cuda'),
    }

    # Run transform twice
    result1 = scale(pc, seed=42)
    result2 = scale(pc, seed=42)

    # Results should be identical
    assert torch.allclose(result1['pos'], result2['pos'])
    assert torch.allclose(result1['feat'], result2['feat'])


def test_scale_empty_pc():
    """Test Scale transform raises assertion error for empty point cloud."""
    scale = Scale(scale_factor=0.1)
    pc = {
        'pos': torch.empty(0, 3),
        'feat': torch.empty(0, 4),
    }

    with pytest.raises(AssertionError) as exc_info:
        scale(pc)
    
    # Check error message
    assert "pc['pos'].shape=torch.Size([0, 3])" in str(exc_info.value)


def test_scale_too_small():
    """Test Scale transform raises error when scale factor is too small."""
    # Create a small point cloud
    pc = {
        'pos': torch.randn(10, 3),
        'feat': torch.randn(10, 4),
    }

    # Scale factor that would result in 0 points (0.1^3 * 10 < 1)
    scale = Scale(scale_factor=0.1)
    
    with pytest.raises(ValueError) as exc_info:
        scale(pc)
    
    # Check error message
    assert "Scale factor 0.1 is too small for point cloud with 10 points" in str(exc_info.value)
    assert "Would result in 0 points after scaling" in str(exc_info.value)
