"""Test ICP point cloud registration method."""
import pytest
import torch
from tests.models.utils.point_cloud_data import generate_point_cloud_data
from .common import get_dummy_input, validate_transformation_output


def get_icp_model(threshold: float = 0.02, max_iterations: int = 50):
    """Get ICP model instance."""
    from models.point_cloud_registration.classic.icp import ICP
    return ICP(threshold=threshold, max_iterations=max_iterations)


def test_icp_initialization():
    """Test ICP initialization."""
    model = get_icp_model(threshold=0.01, max_iterations=100)
    assert model.threshold == 0.01
    assert model.max_iterations == 100


def test_icp_forward_pass_basic():
    """Test basic forward pass."""
    model = get_icp_model()
    model.eval()

    dummy_input = get_dummy_input()

    with torch.no_grad():
        output = model(dummy_input)

    validate_transformation_output(output, batch_size=2)


def test_icp_identical_point_clouds():
    """Test ICP with identical source and target point clouds."""
    model = get_icp_model()
    model.eval()

    # Create identical point clouds
    points = torch.randn(2, 100, 3)
    dummy_input = {
        'src_pc': {'pos': points},
        'tgt_pc': {'pos': points}
    }

    with torch.no_grad():
        output = model(dummy_input)

    # Should return identity transformation
    identity = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    assert torch.allclose(output, identity, atol=1e-3)


def test_icp_different_threshold_values():
    """Test ICP with different threshold values."""
    thresholds = [0.01, 0.02, 0.05, 0.1]

    for threshold in thresholds:
        model = get_icp_model(threshold=threshold)
        dummy_input = get_dummy_input()

        with torch.no_grad():
            output = model(dummy_input)

        validate_transformation_output(output, batch_size=2)


def test_icp_different_max_iterations():
    """Test ICP with different max iteration values."""
    iterations = [10, 20, 50, 100]

    for max_iter in iterations:
        model = get_icp_model(max_iterations=max_iter)
        dummy_input = get_dummy_input()

        with torch.no_grad():
            output = model(dummy_input)

        validate_transformation_output(output, batch_size=2)


def test_icp_different_point_cloud_sizes():
    """Test ICP with different point cloud sizes."""
    model = get_icp_model()
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
def test_icp_different_batch_sizes(batch_size):
    """Test ICP with different batch sizes."""
    model = get_icp_model()
    model.eval()

    dummy_input = generate_point_cloud_data(batch_size=batch_size)

    with torch.no_grad():
        output = model(dummy_input)

    validate_transformation_output(output, batch_size=batch_size)


def test_icp_computational_efficiency():
    """Test that ICP completes in reasonable time."""
    import time

    model = get_icp_model(threshold=0.02, max_iterations=10)

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
    assert computation_time < 5.0, f"ICP took too long: {computation_time:.2f}s"
    assert output is not None
