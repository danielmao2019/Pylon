"""Test RANSAC FPFH point cloud registration method."""
import pytest
import torch
from tests.models.utils.point_cloud_data import generate_point_cloud_data
from .common import get_dummy_input, validate_transformation_output


def get_ransac_fpfh_model(voxel_size: float = 0.05):
    """Get RANSAC FPFH model instance."""
    from models.point_cloud_registration.classic.ransac_fpfh import RANSAC_FPFH
    return RANSAC_FPFH(voxel_size=voxel_size)


def test_ransac_fpfh_initialization():
    """Test RANSAC FPFH initialization."""
    model = get_ransac_fpfh_model(voxel_size=0.1)
    assert model.voxel_size == 0.1


def test_ransac_fpfh_forward_pass_basic():
    """Test basic forward pass."""
    model = get_ransac_fpfh_model()
    model.eval()

    dummy_input = get_dummy_input()

    with torch.no_grad():
        output = model(dummy_input)

    validate_transformation_output(output, batch_size=2)


def test_ransac_fpfh_different_voxel_sizes():
    """Test RANSAC FPFH with different voxel sizes."""
    voxel_sizes = [0.02, 0.05, 0.1, 0.2]

    for voxel_size in voxel_sizes:
        model = get_ransac_fpfh_model(voxel_size=voxel_size)
        dummy_input = get_dummy_input()

        with torch.no_grad():
            output = model(dummy_input)

        validate_transformation_output(output, batch_size=2)


def test_ransac_fpfh_different_voxel_sizes_detailed():
    """Test RANSAC FPFH with additional voxel size values."""
    voxel_sizes = [0.01, 0.03, 0.07, 0.15]

    for voxel_size in voxel_sizes:
        model = get_ransac_fpfh_model(voxel_size=voxel_size)
        dummy_input = get_dummy_input()

        with torch.no_grad():
            output = model(dummy_input)

        validate_transformation_output(output, batch_size=2)


def test_ransac_fpfh_different_point_cloud_sizes():
    """Test RANSAC FPFH with different point cloud sizes."""
    model = get_ransac_fpfh_model()
    model.eval()

    sizes = [50, 100, 500, 1000]

    for size in sizes:
        dummy_input = generate_point_cloud_data(
            batch_size=1, num_points=size, feature_dim=32
        )

        with torch.no_grad():
            output = model(dummy_input)

        validate_transformation_output(output, batch_size=1)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_ransac_fpfh_different_batch_sizes(batch_size):
    """Test RANSAC FPFH with different batch sizes."""
    model = get_ransac_fpfh_model()
    model.eval()

    dummy_input = generate_point_cloud_data(batch_size=batch_size)

    with torch.no_grad():
        output = model(dummy_input)

    validate_transformation_output(output, batch_size=batch_size)


def test_ransac_fpfh_computational_efficiency():
    """Test that RANSAC FPFH completes in reasonable time."""
    import time

    model = get_ransac_fpfh_model(voxel_size=0.1)

    # Test with small point clouds for speed
    dummy_input = generate_point_cloud_data(
        batch_size=1, num_points=50, feature_dim=32
    )

    model.eval()
    start_time = time.time()

    with torch.no_grad():
        output = model(dummy_input)

    end_time = time.time()
    computation_time = end_time - start_time

    # Should complete within reasonable time (5 seconds)
    assert computation_time < 5.0, f"RANSAC FPFH took too long: {computation_time:.2f}s"
    assert output is not None
