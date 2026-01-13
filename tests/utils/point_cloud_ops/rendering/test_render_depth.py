"""Test cases for depth rendering from point clouds."""

import pytest
import torch
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.ops.rendering import render_depth_from_point_cloud


def test_render_depth_basic() -> None:
    """Test basic depth rendering without mask."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],  # Center, depth 1
                [0.5, 0.5, -2.0],  # Upper right, depth 2
                [-0.5, 0.5, -1.5],  # Upper left, depth 1.5
                [0.0, -0.5, -3.0],  # Bottom center, depth 3
            ],
            dtype=torch.float32,
        )
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    resolution = (100, 100)

    depth_map = render_depth_from_point_cloud(
        pc=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        convention="opengl",
        return_mask=False,
    )

    assert depth_map.shape == (100, 100)
    assert depth_map.dtype == torch.float32

    valid_depths = depth_map[depth_map != -1.0]
    assert (valid_depths > 0).all()


def test_render_depth_with_mask() -> None:
    """Test depth rendering with valid mask."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.2, 0.2, -1.5],
                [-0.3, -0.3, -2.0],
            ],
            dtype=torch.float32,
        )
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    resolution = (100, 100)

    depth_map, valid_mask = render_depth_from_point_cloud(
        pc=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        convention="opengl",
        return_mask=True,
    )

    assert depth_map.shape == (100, 100)
    assert valid_mask.shape == (100, 100)
    assert valid_mask.dtype == torch.bool
    assert valid_mask.sum() > 0
    assert valid_mask.sum() < 100 * 100
    assert (depth_map[valid_mask] > 0).all()
    assert (depth_map[~valid_mask] == -1.0).all()


def test_render_depth_sorting() -> None:
    """Test that closer points overwrite farther ones."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -3.0],  # Farther point (depth 3)
                [0.0, 0.0, -1.0],  # Closer point (depth 1, should overwrite)
            ],
            dtype=torch.float32,
        )
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    resolution = (100, 100)

    depth_map = render_depth_from_point_cloud(
        pc=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
    )

    valid_depths = depth_map[depth_map != -1.0]
    if len(valid_depths) > 0:
        assert valid_depths.min() < 1.5


def test_render_depth_custom_ignore_value() -> None:
    """Test using custom ignore value for empty pixels."""
    pc_data = PointCloud(xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32))

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    custom_ignore = -999.0

    depth_map = render_depth_from_point_cloud(
        pc=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        ignore_value=custom_ignore,
    )

    background_pixels = depth_map == custom_ignore
    assert background_pixels.sum() > 100 * 100 * 0.9


def test_render_depth_points_behind_camera() -> None:
    """Test that points behind camera are filtered out."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],  # Behind camera (positive Z in OpenGL)
                [0.0, 0.0, -1.0],  # In front of camera
            ],
            dtype=torch.float32,
        )
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    depth_map, valid_mask = render_depth_from_point_cloud(
        pc=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        return_mask=True,
    )

    assert valid_mask.sum() >= 1
    assert (depth_map[valid_mask] > 0).all()


def test_render_depth_multiple_points_per_pixel() -> None:
    """Test that multiple points projecting to same pixel are handled correctly."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.01, 0.0, -2.0],
                [0.0, 0.01, -1.5],
                [-0.01, 0.0, -3.0],
            ],
            dtype=torch.float32,
        )
    )

    camera_intrinsics = torch.tensor(
        [
            [1000.0, 0.0, 50.0],
            [0.0, 1000.0, 50.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    depth_map = render_depth_from_point_cloud(
        pc=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
    )

    center_region = depth_map[48:52, 48:52]
    valid_depths = center_region[center_region != -1.0]
    if len(valid_depths) > 0:
        assert valid_depths.min() < 1.5


def test_render_depth_intrinsics_scaling() -> None:
    """Test that intrinsics are properly scaled for different resolutions."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.5, 0.5, -2.0],
            ],
            dtype=torch.float32,
        )
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    depth_map_small = render_depth_from_point_cloud(
        pc=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(50, 50),
    )

    depth_map_large = render_depth_from_point_cloud(
        pc=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(200, 200),
    )

    assert depth_map_small.shape == (50, 50)
    assert depth_map_large.shape == (200, 200)
    assert (depth_map_small != -1.0).any()
    assert (depth_map_large != -1.0).any()


def test_render_depth_invalid_inputs() -> None:
    """Test various invalid input conditions."""
    valid_pc_data = PointCloud(xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32))
    valid_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    valid_extrinsics = torch.eye(4, dtype=torch.float32)

    with pytest.raises(AssertionError):
        render_depth_from_point_cloud(
            pc="not a point cloud",
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100),
        )

    with pytest.raises(AssertionError):
        render_depth_from_point_cloud(
            pc=valid_pc_data,
            camera_intrinsics=torch.eye(4),
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100),
        )

    with pytest.raises(AssertionError):
        render_depth_from_point_cloud(
            pc=valid_pc_data,
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=torch.eye(3),
            resolution=(100, 100),
        )

    with pytest.raises(AssertionError):
        render_depth_from_point_cloud(
            pc=valid_pc_data,
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=valid_extrinsics,
            resolution=(0, 100),
        )
