"""Test cases for normal-map rendering from point clouds."""

from typing import Tuple

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.three_d.point_cloud.render import render_normal_from_point_cloud_3d


def test_render_normal_lands_on_the_pixel_of_its_own_point() -> None:
    """Test that each rendered pixel carries the normal of the point that projected onto it."""
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
            'normals': torch.tensor(
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

    normal_map = render_normal_from_point_cloud_3d(
        pc=pc_data,
        camera=camera,
        resolution=resolution,
    )

    # The identity OpenGL pose reaches OpenCV as the world-to-camera rotation
    # diag(1, -1, -1), so a world normal lands in the camera frame with its y
    # and z components negated.
    camera_normals = pc_data.normals * torch.tensor(
        [1.0, -1.0, -1.0], dtype=torch.float32
    )
    pixels = [(50, 50), (43, 62), (56, 37)]
    for point_index, (row, column) in enumerate(pixels):
        assert (normal_map[:, row, column] == camera_normals[point_index]).all(), (
            "Each rendered pixel must carry the normal of the point that projected onto it. "
            f"{point_index=} {row=} {column=} "
            f"{normal_map[:, row, column]=} {camera_normals[point_index]=}"
        )


def test_render_normal_ignores_the_points_that_culled_out() -> None:
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
            'normals': torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        },
    )

    camera = _build_camera(focal=100.0, principal_point=50.0)
    resolution: Tuple[int, int] = (100, 100)

    normal_map = render_normal_from_point_cloud_3d(
        pc=pc_data,
        camera=camera,
        resolution=resolution,
    )

    painted = (normal_map != 0.0).any(dim=0)
    expected_painted = torch.zeros(resolution, dtype=torch.bool)
    for row, column in [(50, 50), (43, 62), (56, 37)]:
        expected_painted[row, column] = True
    assert torch.equal(painted, expected_painted), (
        "Exactly the three survivors' pixels must be painted. "
        f"{painted.sum()=} {expected_painted.sum()=} {painted.nonzero().tolist()=}"
    )

    # The identity OpenGL pose reaches OpenCV as the world-to-camera rotation
    # diag(1, -1, -1), so a world normal lands in the camera frame with its y
    # and z components negated.
    camera_normals = pc_data.normals * torch.tensor(
        [1.0, -1.0, -1.0], dtype=torch.float32
    )
    for culled_index in (3, 4):
        culled_normal = camera_normals[culled_index].reshape(3, 1, 1)
        assert not (normal_map == culled_normal).all(dim=0).any(), (
            "No pixel may carry the normal of a point that culled out. "
            f"{culled_index=} {camera_normals[culled_index]=} "
            f"{(normal_map == culled_normal).all(dim=0).nonzero().tolist()=}"
        )


def test_render_normal_takes_the_nearest_point_where_two_share_a_pixel() -> None:
    """Test that two points on one ray paint the nearer one's normal."""
    pc_data = PointCloud(
        xyz=torch.tensor(
            [
                [0.0, 0.0, -3.0],  # far point, first in pc.xyz order
                [0.0, 0.0, -1.0],  # near point, second in pc.xyz order
            ],
            dtype=torch.float32,
        ),
        data={
            'normals': torch.tensor(
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

    normal_map = render_normal_from_point_cloud_3d(
        pc=pc_data,
        camera=camera,
        resolution=resolution,
    )

    # The identity OpenGL pose reaches OpenCV as the world-to-camera rotation
    # diag(1, -1, -1), so a world normal lands in the camera frame with its y
    # and z components negated.
    camera_normals = pc_data.normals * torch.tensor(
        [1.0, -1.0, -1.0], dtype=torch.float32
    )
    assert (normal_map[:, 50, 50] == camera_normals[1]).all(), (
        "The pixel two points share must carry the near point's normal. "
        f"{normal_map[:, 50, 50]=} {camera_normals[1]=} {camera_normals[0]=}"
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
