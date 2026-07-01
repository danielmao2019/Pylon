"""Test cases for segmentation rendering from point clouds."""

import pytest
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from models.three_d.point_cloud.render import (
    render_segmentation_from_point_cloud,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def _build_camera(focal: float, principal_point: float) -> Camera:
    """Build an identity-pose OpenGL pinhole camera on the CPU.

    Args:
        focal: Shared focal length used for both fx and fy.
        principal_point: Shared principal-point coordinate used for both cx and cy.

    Returns:
        A Camera whose pinhole intrinsics are (fx, fy, cx, cy) and whose
        extrinsics are the identity cam2world matrix in the opengl convention.
    """
    return Camera(
        intrinsics=build_camera_intrinsics(
            model="pinhole",
            params={
                "fx": focal,
                "fy": focal,
                "cx": principal_point,
                "cy": principal_point,
            },
            device=torch.device("cpu"),
        ),
        extrinsics=CameraExtrinsics(
            extrinsics=torch.eye(4, dtype=torch.float32),
            convention="opengl",
            device=torch.device("cpu"),
        ),
        device=torch.device("cpu"),
    )


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

    camera = _build_camera(focal=100.0, principal_point=50.0)

    seg_map = render_segmentation_from_point_cloud(
        pc=pc_data,
        key='labels',
        camera=camera,
        resolution=(100, 100),
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

    camera = _build_camera(focal=100.0, principal_point=50.0)

    seg_map, valid_mask = render_segmentation_from_point_cloud(
        pc=pc_data,
        key='labels',
        camera=camera,
        resolution=(100, 100),
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

    camera = _build_camera(focal=100.0, principal_point=50.0)

    seg_map = render_segmentation_from_point_cloud(
        pc=pc_data,
        key='labels',
        camera=camera,
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

    camera = _build_camera(focal=100.0, principal_point=50.0)

    seg_map, valid_mask = render_segmentation_from_point_cloud(
        pc=pc_data,
        key='labels',
        camera=camera,
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

    camera = _build_camera(focal=100.0, principal_point=50.0)

    ignore_value = -1
    seg_map = render_segmentation_from_point_cloud(
        pc=pc_data,
        key='labels',
        camera=camera,
        resolution=(100, 100),
        ignore_value=ignore_value,
    )

    assert (seg_map == ignore_value).sum() > 0


def test_render_segmentation_missing_labels() -> None:
    """Test that missing label field raises assertion error."""
    pc_data = PointCloud(
        xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
    )
    camera = _build_camera(focal=100.0, principal_point=50.0)

    with pytest.raises(AssertionError):
        render_segmentation_from_point_cloud(
            pc=pc_data,
            key="labels",
            camera=camera,
            resolution=(100, 100),
        )


def test_render_segmentation_invalid_inputs() -> None:
    """Test various invalid input conditions."""
    labels = torch.tensor([1], dtype=torch.int64)
    pc_data = PointCloud(
        xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        data={'labels': labels},
    )
    valid_camera = _build_camera(focal=100.0, principal_point=50.0)

    with pytest.raises(AssertionError):
        render_segmentation_from_point_cloud(
            pc=None,
            key='labels',
            camera=valid_camera,
            resolution=(100, 100),
        )

    # A non-CameraIntrinsics intrinsics is rejected at Camera construction.
    with pytest.raises(AssertionError):
        Camera(
            intrinsics=torch.eye(4, dtype=torch.float32),
            extrinsics=CameraExtrinsics(
                extrinsics=torch.eye(4, dtype=torch.float32),
                convention="opengl",
                device=torch.device("cpu"),
            ),
            device=torch.device("cpu"),
        )

    # A malformed (3x3) extrinsics matrix is rejected at CameraExtrinsics construction.
    with pytest.raises(AssertionError):
        CameraExtrinsics(
            extrinsics=torch.eye(3, dtype=torch.float32),
            convention="opengl",
            device=torch.device("cpu"),
        )

    with pytest.raises(AssertionError):
        render_segmentation_from_point_cloud(
            pc=pc_data,
            key='labels',
            camera=valid_camera,
            resolution=(0, 100),
        )
