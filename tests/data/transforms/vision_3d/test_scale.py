import pytest
import torch
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.transforms.vision_3d.scale import Scale

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_pc():
    """Create a sample PointCloud for testing."""
    num_points = 1000
    xyz = torch.randn(size=(num_points, 3), device=DEVICE)
    return PointCloud(
        xyz=xyz,
        data={
            'feat': torch.randn(size=(num_points, 4), device=DEVICE),
            'normal': torch.randn(size=(num_points, 3), device=DEVICE),
        },
    )


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

    assert isinstance(result, PointCloud)
    assert result.field_names() == ('xyz', 'feat', 'normal')

    # Check if shapes are consistent
    expected_num_points = int(sample_pc.num_points * (0.5 ** 3))
    assert result.xyz.shape == (expected_num_points, 3)
    assert result.feat.shape == (expected_num_points, 4)
    assert result.normal.shape == (expected_num_points, 3)

    # Get expected indices for deterministic comparison
    generator = torch.Generator(device=sample_pc.xyz.device)
    generator.manual_seed(42)
    indices = torch.randperm(sample_pc.num_points, device=sample_pc.xyz.device, generator=generator)[:expected_num_points]

    # Check if XYZ coordinates are scaled
    assert torch.allclose(result.xyz, sample_pc.xyz[indices] * 0.5)
    assert torch.allclose(result.feat, sample_pc.feat[indices])
    assert torch.allclose(result.normal, sample_pc.normal[indices])


def test_scale_invalid_input():
    """Test Scale transform with invalid inputs."""
    scale = Scale(scale_factor=0.1)

    with pytest.raises(AssertionError):
        scale("not a PointCloud")


def test_scale_deterministic():
    """Test if Scale transform produces consistent results with same input."""
    scale = Scale(scale_factor=0.1)
    pc = PointCloud(
        xyz=torch.randn(size=(1000, 3), device=DEVICE),
        data={'feat': torch.randn(size=(1000, 4), device=DEVICE)},
    )

    # Run transform twice
    result1 = scale(pc, seed=42)
    result2 = scale(pc, seed=42)

    # Results should be identical
    assert torch.allclose(result1.xyz, result2.xyz)
    assert torch.allclose(result1.feat, result2.feat)


def test_scale_too_small():
    """Test Scale transform raises error when scale factor is too small."""
    # Create a small point cloud
    pc = PointCloud(
        xyz=torch.randn(10, 3),
        data={'feat': torch.randn(10, 4)},
    )

    # Scale factor that would result in 0 points (0.1^3 * 10 < 1)
    scale = Scale(scale_factor=0.1)

    with pytest.raises(ValueError) as exc_info:
        scale(pc)

    assert "Scale factor 0.1 is too small for point cloud with 10 points" in str(
        exc_info.value
    )
    assert "Would result in 0 points after scaling" in str(exc_info.value)
