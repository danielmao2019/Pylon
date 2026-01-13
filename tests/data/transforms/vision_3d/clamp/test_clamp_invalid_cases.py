import pytest
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.transforms.vision_3d.clamp import Clamp


def test_clamp_invalid_initialization():
    """Test Clamp transform initialization with invalid parameters."""
    # Test invalid max_points type
    with pytest.raises(AssertionError):
        Clamp(max_points="invalid")

    with pytest.raises(AssertionError):
        Clamp(max_points=1.5)

    with pytest.raises(AssertionError):
        Clamp(max_points=None)

    # Test invalid max_points values
    with pytest.raises(AssertionError):
        Clamp(max_points=0)

    with pytest.raises(AssertionError):
        Clamp(max_points=-1)

    with pytest.raises(AssertionError):
        Clamp(max_points=-100)


def test_clamp_invalid_input_types():
    """Test Clamp transform with invalid input types."""
    clamp = Clamp(max_points=100)

    # Test with non-PointCloud input
    with pytest.raises(AssertionError):
        clamp("not a dict")

    with pytest.raises(AssertionError):
        clamp(123)

    with pytest.raises(AssertionError):
        clamp([1, 2, 3])

    # Test with no arguments
    with pytest.raises(AssertionError, match="len\\(args\\)=0"):
        clamp()


def test_clamp_invalid_point_cloud_structure():
    """Test Clamp transform with invalid point cloud structure."""
    clamp = Clamp(max_points=100)

    # Test with tensor instead of PointCloud
    with pytest.raises(AssertionError):
        clamp(torch.randn(10, 3, dtype=torch.float32))


def test_clamp_inconsistent_point_counts():
    """Test Clamp transform with point clouds having different point counts."""
    clamp = Clamp(max_points=50)

    data1 = {
        'xyz': torch.randn(100, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.randn(100, 4, dtype=torch.float32, device='cuda'),
    }
    data2 = {
        'xyz': torch.randn(200, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.randn(200, 4, dtype=torch.float32, device='cuda'),
    }
    pc1 = PointCloud(data=data1)
    pc2 = PointCloud(data=data2)

    with pytest.raises(ValueError, match="All point clouds must have the same number of points"):
        clamp(pc1, pc2, seed=42)


def test_clamp_inconsistent_devices():
    """Test Clamp transform with point clouds on different devices."""
    clamp = Clamp(max_points=50)

    data1 = {
        'xyz': torch.randn(100, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.randn(100, 4, dtype=torch.float32, device='cuda'),
    }
    data2 = {
        'xyz': torch.randn(100, 3, dtype=torch.float32, device='cpu'),
        'feat': torch.randn(100, 4, dtype=torch.float32, device='cpu'),
    }
    pc1 = PointCloud(data=data1)
    pc2 = PointCloud(data=data2)

    with pytest.raises(ValueError, match="All point clouds must be on the same device"):
        clamp(pc1, pc2, seed=42)


def test_clamp_wrong_xyz_dimensions():
    """Test Clamp transform with wrong xyz tensor dimensions."""
    clamp = Clamp(max_points=100)

    with pytest.raises(AssertionError):
        PointCloud(xyz=torch.randn(10, dtype=torch.float32), data={'feat': torch.randn(10, 4, dtype=torch.float32)})

    with pytest.raises(AssertionError):
        PointCloud(xyz=torch.randn(10, 2, dtype=torch.float32), data={'feat': torch.randn(10, 4, dtype=torch.float32)})


def test_clamp_inconsistent_shapes_within_pc():
    """Test Clamp transform with inconsistent shapes within single point cloud."""
    clamp = Clamp(max_points=100)

    with pytest.raises(AssertionError):
        PointCloud(xyz=torch.randn(10, 3, dtype=torch.float32), data={'feat': torch.randn(8, 4, dtype=torch.float32)})


def test_clamp_wrong_tensor_type():
    """Test Clamp transform with non-tensor values."""
    clamp = Clamp(max_points=100)

    with pytest.raises(AssertionError):
        PointCloud(xyz=[[1, 2, 3], [4, 5, 6]], data={'feat': torch.randn(2, 4, dtype=torch.float32)})


@pytest.mark.parametrize("invalid_input", [
    None,
    [],
    (),
    42,
    "string",
    torch.tensor([1, 2, 3]),
])
def test_clamp_various_invalid_inputs(invalid_input):
    """Test Clamp transform with various invalid input types."""
    clamp = Clamp(max_points=100)

    with pytest.raises(AssertionError):
        clamp(invalid_input)


def test_clamp_mixed_valid_invalid_multi_args():
    """Test multi-arg Clamp with mix of valid and invalid point clouds."""
    clamp = Clamp(max_points=50)

    valid_pc = PointCloud(data={
        'xyz': torch.randn(100, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.randn(100, 4, dtype=torch.float32, device='cuda'),
    })

    with pytest.raises(AssertionError):
        clamp(valid_pc, torch.randn(100, 2, dtype=torch.float32, device='cuda'), seed=42)


def test_clamp_multi_args_different_point_counts():
    """Test multi-arg Clamp where point clouds have different point counts."""
    clamp = Clamp(max_points=50)

    pc1 = PointCloud(data={
        'xyz': torch.randn(100, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.randn(100, 4, dtype=torch.float32, device='cuda'),
    })

    pc2 = PointCloud(data={
        'xyz': torch.randn(200, 3, dtype=torch.float32, device='cuda'),
        'feat': torch.randn(200, 4, dtype=torch.float32, device='cuda'),
    })

    # This should fail due to different point counts (100 vs 200)
    with pytest.raises(ValueError, match="All point clouds must have the same number of points"):
        clamp(pc1, pc2, seed=42)
