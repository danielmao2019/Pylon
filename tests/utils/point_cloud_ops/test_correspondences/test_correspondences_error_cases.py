import pytest
import torch

from data.structures.three_d.point_cloud.ops.correspondences import get_correspondences
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def test_get_correspondences_invalid_source_type():
    """Test that passing a non-PointCloud source raises an error."""
    target_pc = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    radius = 1.0

    with pytest.raises(AssertionError):
        get_correspondences(source=None, target=target_pc, transform=None, radius=radius)


def test_get_correspondences_invalid_target_type():
    """Test that passing a non-PointCloud target raises an error."""
    source_pc = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    radius = 1.0

    with pytest.raises(AssertionError):
        get_correspondences(source=source_pc, target=None, transform=None, radius=radius)


def test_get_correspondences_mismatched_devices():
    """Test that error is raised when source and target are on different devices."""
    source_pc = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    # This test can only run if CUDA is available
    if torch.cuda.is_available():
        target_pc = PointCloud(
            xyz=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
                device='cuda',
            )
        )

        radius = 1.0

        with pytest.raises(AssertionError, match="src_points.device="):
            get_correspondences(source_pc, target_pc, None, radius)


def test_get_correspondences_invalid_transform_shape():
    """Test that error is raised for invalid transform shape."""
    source_pc = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    target_pc = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    # Invalid transform shape (should be 4x4)
    invalid_transform = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    radius = 1.0

    with pytest.raises(AssertionError, match="Invalid transform shape"):
        get_correspondences(source_pc, target_pc, invalid_transform, radius)


def test_get_correspondences_negative_radius():
    """Test that error is raised for negative radius."""
    source_pc = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    target_pc = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    negative_radius = -0.5

    with pytest.raises(AssertionError, match="radius must be positive number"):
        get_correspondences(source_pc, target_pc, None, negative_radius)


def test_get_correspondences_zero_radius():
    """Test that error is raised for zero radius."""
    source_pc = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    target_pc = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    zero_radius = 0.0

    with pytest.raises(AssertionError, match="radius must be positive number"):
        get_correspondences(source_pc, target_pc, None, zero_radius)
