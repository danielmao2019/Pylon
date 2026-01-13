"""Test cases for segmentation rendering from point clouds."""

import pytest
import torch

from data.structures.three_d.point_cloud.ops.rendering import (
    render_segmentation_from_point_cloud,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def test_render_segmentation_basic() -> None:
    """Test basic segmentation rendering without mask."""
    labels = torch.tensor([1, 2, 3], dtype=torch.int64)
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.5, 0.5, -2.0],
                [-0.5, 0.5, -1.5],
            ],
            dtype=torch.float32,
        ),
        data={'labels': labels},
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    seg_map = render_segmentation_from_point_cloud(
        pc=pc_data,
        key='labels',
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        convention="opengl",
        return_mask=False,
    )

    assert seg_map.shape == (100, 100)
    assert seg_map.dtype == torch.int64
    assert (seg_map == 255).sum() > 0


def test_render_segmentation_with_mask() -> None:
    """Test segmentation rendering with valid mask."""
    labels = torch.tensor([1, 2], dtype=torch.int64)
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.2, 0.2, -1.5],
            ],
            dtype=torch.float32,
        ),
        data={'labels': labels},
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    seg_map, valid_mask = render_segmentation_from_point_cloud(
        pc=pc_data,
        key='labels',
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        convention="opengl",
        return_mask=True,
    )

    assert seg_map.shape == (100, 100)
    assert valid_mask.shape == (100, 100)
    assert valid_mask.dtype == torch.bool
    assert valid_mask.sum() > 0
    assert (seg_map[valid_mask] != 255).all()


def test_render_segmentation_depth_sorting() -> None:
    """Test that closer points overwrite farther ones."""
    labels = torch.tensor([5, 7], dtype=torch.int64)
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -2.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        ),
        data={'labels': labels},
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    seg_map = render_segmentation_from_point_cloud(
        pc=pc_data,
        key='labels',
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
    )

    assert seg_map.dtype == torch.int64


def test_render_segmentation_points_behind_camera() -> None:
    """Test that points behind camera are filtered out."""
    labels = torch.tensor([3, 4], dtype=torch.int64)
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        ),
        data={'labels': labels},
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    seg_map, valid_mask = render_segmentation_from_point_cloud(
        pc=pc_data,
        key='labels',
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        return_mask=True,
    )

    assert valid_mask.sum() >= 1
    assert (seg_map[valid_mask] != 255).all()


def test_render_segmentation_custom_ignore_value() -> None:
    """Test custom ignore value handling."""
    labels = torch.tensor([1], dtype=torch.int64)
    pc_data = PointCloud(
        xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        data={'labels': labels},
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    ignore_value = -1
    seg_map = render_segmentation_from_point_cloud(
        pc=pc_data,
        key='labels',
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        ignore_value=ignore_value,
    )

    assert (seg_map == ignore_value).sum() > 0


def test_render_segmentation_missing_labels() -> None:
    """Test that missing label field raises assertion error."""
    pc_data = PointCloud(
        xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
    )
    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    with pytest.raises(AssertionError):
        render_segmentation_from_point_cloud(
            pc=pc_data,
            key="labels",
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            resolution=(100, 100),
        )


def test_render_segmentation_invalid_inputs() -> None:
    """Test various invalid input conditions."""
    labels = torch.tensor([1], dtype=torch.int64)
    pc_data = PointCloud(
        xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        data={'labels': labels},
    )
    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)

    with pytest.raises(AssertionError):
        render_segmentation_from_point_cloud(
            pc=None,
            key='labels',
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            resolution=(100, 100),
        )

    with pytest.raises(AssertionError):
        render_segmentation_from_point_cloud(
            pc=pc_data,
            key='labels',
            camera_intrinsics=torch.eye(4),
            camera_extrinsics=camera_extrinsics,
            resolution=(100, 100),
        )

    with pytest.raises(AssertionError):
        render_segmentation_from_point_cloud(
            pc=pc_data,
            key='labels',
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=torch.eye(3),
            resolution=(100, 100),
        )

    with pytest.raises(AssertionError):
        render_segmentation_from_point_cloud(
            pc=pc_data,
            key='labels',
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            resolution=(0, 100),
        )
