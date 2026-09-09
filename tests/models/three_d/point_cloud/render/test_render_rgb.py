"""Test cases for RGB rendering from point clouds."""

from typing import Tuple

import pytest
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.three_d.point_cloud.render import render_rgb_from_point_cloud


def test_render_rgb_lands_on_the_pixel_of_its_own_point() -> None:
    """Test that each rendered pixel carries the colour of the point that projected onto it."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],  # depth 1, projects to (row 50, col 50)
                [0.25, 0.125, -2.0],  # depth 2, projects to (row 43, col 62)
                [-0.5, -0.25, -4.0],  # depth 4, projects to (row 56, col 37)
            ],
            dtype=torch.float32,
        ),
        data={
            'rgb': torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        },
    )

    camera = _build_camera(focal=100.0, principal_point=50.0)
    resolution: Tuple[int, int] = (100, 100)

    rgb_image = render_rgb_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=resolution,
    )

    pixels = [(50, 50), (43, 62), (56, 37)]
    for point_index, (row, column) in enumerate(pixels):
        assert torch.equal(rgb_image[:, row, column], pc_data.rgb[point_index]), (
            "Each rendered pixel must carry the colour of the point that projected onto it. "
            f"{point_index=} {row=} {column=} "
            f"{rgb_image[:, row, column]=} {pc_data.rgb[point_index]=}"
        )


def test_render_rgb_ignores_the_points_that_culled_out() -> None:
    """Test that a cloud of several survivors and several culled points renders only the survivors."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -1.0],  # depth 1, projects to (row 50, col 50)
                [0.25, 0.125, -2.0],  # depth 2, projects to (row 43, col 62)
                [-0.5, -0.25, -4.0],  # depth 4, projects to (row 56, col 37)
                [10.0, 0.0, -1.0],  # projects far right of the image bounds
                [0.0, 10.0, -1.0],  # projects far above the image bounds
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
                    [0.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            )
        },
    )

    camera = _build_camera(focal=100.0, principal_point=50.0)
    resolution: Tuple[int, int] = (100, 100)

    rgb_image = render_rgb_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=resolution,
    )

    painted = (rgb_image != 0.0).any(dim=0)
    expected_painted = torch.zeros(resolution, dtype=torch.bool)
    for row, column in [(50, 50), (43, 62), (56, 37)]:
        expected_painted[row, column] = True
    assert torch.equal(painted, expected_painted), (
        "Exactly the three survivors' pixels must be painted. "
        f"{painted.sum()=} {expected_painted.sum()=} {painted.nonzero().tolist()=}"
    )

    for culled_index in (3, 4):
        culled_color = pc_data.rgb[culled_index].reshape(3, 1, 1)
        assert not (rgb_image == culled_color).all(dim=0).any(), (
            "No pixel may carry the colour of a point that culled out. "
            f"{culled_index=} {pc_data.rgb[culled_index]=} "
            f"{(rgb_image == culled_color).all(dim=0).nonzero().tolist()=}"
        )


def test_render_rgb_takes_the_nearest_point_where_two_share_a_pixel() -> None:
    """Test that two points on one ray paint the nearer one's colour."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -3.0],  # far point, first in pc.xyz order
                [0.0, 0.0, -1.0],  # near point, second in pc.xyz order
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

    camera = _build_camera(focal=100.0, principal_point=50.0)
    resolution: Tuple[int, int] = (100, 100)

    rgb_image = render_rgb_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=resolution,
    )

    assert torch.equal(rgb_image[:, 50, 50], pc_data.rgb[1]), (
        "The pixel two points share must carry the near point's colour. "
        f"{rgb_image[:, 50, 50]=} {pc_data.rgb[1]=} {pc_data.rgb[0]=}"
    )


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

    camera = _build_camera(focal=100.0, principal_point=50.0)

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

    camera = _build_camera(focal=100.0, principal_point=50.0)

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

    camera = _build_camera(focal=100.0, principal_point=50.0)

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

    camera = _build_camera(focal=100.0, principal_point=50.0)

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

    camera = _build_camera(focal=100.0, principal_point=50.0)

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

    camera = _build_camera(focal=100.0, principal_point=50.0)

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
    camera = _build_camera(focal=100.0, principal_point=50.0)

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
    camera = _build_camera(focal=100.0, principal_point=50.0)

    with pytest.raises(AssertionError):
        render_rgb_from_point_cloud(
            pc=None,
            camera=camera,
            resolution=(100, 100),
        )

    # A malformed (3x3) extrinsics matrix is rejected at CameraExtrinsics construction.
    with pytest.raises(AssertionError):
        CameraExtrinsics(
            extrinsics=torch.eye(3, dtype=torch.float32),
            extr_convention="opengl",
            device=torch.device("cpu"),
        )

    # A non-CameraIntrinsics intrinsics is rejected at Camera construction.
    with pytest.raises(AssertionError):
        Camera(
            intrinsics=torch.eye(4, dtype=torch.float32),
            extrinsics=CameraExtrinsics(
                extrinsics=torch.eye(4, dtype=torch.float32),
                extr_convention="opengl",
                device=torch.device("cpu"),
            ),
            device=torch.device("cpu"),
        )

    with pytest.raises(AssertionError):
        render_rgb_from_point_cloud(
            pc=pc_data,
            camera=camera,
            resolution=(0, 100),
        )


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
                "h": int(round(2.0 * principal_point)),
                "w": int(round(2.0 * principal_point)),
            },
            intr_convention="standard",
            device=torch.device("cpu"),
        ),
        extrinsics=CameraExtrinsics(
            extrinsics=torch.eye(4, dtype=torch.float32),
            extr_convention="opengl",
            device=torch.device("cpu"),
        ),
        device=torch.device("cpu"),
    )
