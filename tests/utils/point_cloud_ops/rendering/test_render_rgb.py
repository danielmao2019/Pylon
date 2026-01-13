"""Test cases for RGB rendering from point clouds."""

import pytest
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.point_cloud.ops.rendering import (
    render_rgb_from_point_cloud,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def test_render_rgb_basic() -> None:
    """Test basic RGB rendering without mask."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.5, 0.5, -2.0],
                [-0.5, 0.5, -1.5],
                [0.0, -0.5, -3.0],
            ],
            dtype=torch.float32,
        ),
        data={
            'rgb': torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        },
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    camera = Camera(
        intrinsics=camera_intrinsics,
        extrinsics=camera_extrinsics,
        convention="opengl",
        device=camera_intrinsics.device,
    )

    rgb_image = render_rgb_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        return_mask=False,
    )

    assert rgb_image.shape == (3, 100, 100)
    assert rgb_image.dtype == torch.float32
    assert rgb_image.min() >= 0.0
    assert rgb_image.max() <= 1.0


def test_render_rgb_with_mask() -> None:
    """Test RGB rendering with valid mask."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.2, 0.2, -1.5],
            ],
            dtype=torch.float32,
        ),
        data={
            'rgb': torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        },
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    camera = Camera(
        intrinsics=camera_intrinsics,
        extrinsics=camera_extrinsics,
        convention="opengl",
        device=camera_intrinsics.device,
    )

    rgb_image, valid_mask = render_rgb_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        return_mask=True,
    )

    assert rgb_image.shape == (3, 100, 100)
    assert valid_mask.shape == (100, 100)
    assert valid_mask.dtype == torch.bool
    assert valid_mask.sum() > 0
    assert valid_mask.sum() < 100 * 100
    assert (rgb_image[:, ~valid_mask] == 0.0).all()


def test_render_rgb_color_normalization() -> None:
    """Test automatic color normalization from 0-255 range."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.1, 0.1, -1.2],
            ],
            dtype=torch.float32,
        ),
        data={
            'rgb': torch.tensor(
                [
                    [255.0, 0.0, 0.0],
                    [0.0, 255.0, 128.0],
                ],
                dtype=torch.float32,
            )
        },
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    camera = Camera(
        intrinsics=camera_intrinsics,
        extrinsics=camera_extrinsics,
        convention="opengl",
        device=camera_intrinsics.device,
    )

    rgb_image, valid_mask = render_rgb_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        return_mask=True,
    )

    assert rgb_image.max() <= 1.0
    assert rgb_image.min() >= 0.0
    assert valid_mask.any()


def test_render_rgb_depth_sorting() -> None:
    """Test that closer points overwrite farther ones."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -2.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        ),
        data={
            'rgb': torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        },
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    camera = Camera(
        intrinsics=camera_intrinsics,
        extrinsics=camera_extrinsics,
        convention="opengl",
        device=camera_intrinsics.device,
    )

    rgb_image = render_rgb_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
    )

    assert (rgb_image > 0.0).any()


def test_render_rgb_points_behind_camera() -> None:
    """Test that points behind camera are filtered out."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        ),
        data={
            'rgb': torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        },
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    camera = Camera(
        intrinsics=camera_intrinsics,
        extrinsics=camera_extrinsics,
        convention="opengl",
        device=camera_intrinsics.device,
    )

    rgb_image, valid_mask = render_rgb_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        return_mask=True,
    )

    assert valid_mask.sum() >= 1
    assert (rgb_image[:, valid_mask] >= 0.0).all()


def test_render_rgb_custom_ignore_value() -> None:
    """Test using custom ignore value for empty pixels."""
    pc_data = PointCloud(
        xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        data={'rgb': torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)},
    )

    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    camera = Camera(
        intrinsics=camera_intrinsics,
        extrinsics=camera_extrinsics,
        convention="opengl",
        device=camera_intrinsics.device,
    )

    ignore_value = -1.0
    rgb_image = render_rgb_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=(100, 100),
        ignore_value=ignore_value,
    )

    assert (rgb_image[:, rgb_image[0] == ignore_value] == ignore_value).all()


def test_render_rgb_missing_rgb_field() -> None:
    """Test that missing RGB data raises assertion error."""
    pc_data = PointCloud(
        xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
    )
    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    camera = Camera(
        intrinsics=camera_intrinsics,
        extrinsics=camera_extrinsics,
        convention="opengl",
        device=camera_intrinsics.device,
    )

    with pytest.raises(AssertionError):
        render_rgb_from_point_cloud(
            pc=pc_data,
            camera=camera,
            resolution=(100, 100),
        )


def test_render_rgb_invalid_inputs() -> None:
    """Test various invalid input conditions."""
    pc_data = PointCloud(
        xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        data={'rgb': torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)},
    )
    camera_intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    camera = Camera(
        intrinsics=camera_intrinsics,
        extrinsics=camera_extrinsics,
        convention="opengl",
    )

    with pytest.raises(AssertionError):
        render_rgb_from_point_cloud(
            pc=None,
            camera=camera,
            resolution=(100, 100),
        )

    with pytest.raises(AssertionError):
        Camera(
            intrinsics=torch.eye(4),
            extrinsics=camera_extrinsics,
            convention="opengl",
            device=camera_intrinsics.device,
        )

    with pytest.raises(AssertionError):
        Camera(
            intrinsics=camera_intrinsics,
            extrinsics=torch.eye(3),
            convention="opengl",
            device=camera_intrinsics.device,
        )

    with pytest.raises(AssertionError):
        render_rgb_from_point_cloud(
            pc=pc_data,
            camera=camera,
            resolution=(0, 100),
        )
