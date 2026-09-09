"""Test cases for the circular kernel offsets that dilate a rendered point into a disc."""

import math
from typing import Dict, Set, Tuple

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import (
    create_circular_kernel_offsets,
)
from models.three_d.point_cloud.render.render_depth import render_depth_from_point_cloud

POINT_SIZES: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)


def test_create_circular_kernel_offsets_disc_is_centred() -> None:
    """Test that the kernel offsets are closed under negation, so the disc is centred on the point."""
    for point_size in POINT_SIZES:
        offsets = _to_offset_set(
            create_circular_kernel_offsets(
                point_size=point_size, device=torch.device("cpu")
            )
        )

        unmatched = {(y, x) for y, x in offsets if (-y, -x) not in offsets}
        assert not unmatched, (
            "Every kernel offset must have its negation in the kernel, otherwise the disc "
            "reaches farther on one side of the point than on the other. "
            f"{point_size=} {sorted(unmatched)=} {sorted(offsets)=}"
        )


def test_create_circular_kernel_offsets_membership_is_the_radius_rule() -> None:
    """Test that the kernel holds exactly the integer cells whose centre lies within point_size / 2 of the origin."""
    for point_size in POINT_SIZES:
        offsets = _to_offset_set(
            create_circular_kernel_offsets(
                point_size=point_size, device=torch.device("cpu")
            )
        )

        kernel_radius = point_size / 2.0
        search_reach = math.ceil(point_size) + 1
        expected = {
            (y, x)
            for y in range(-search_reach, search_reach + 1)
            for x in range(-search_reach, search_reach + 1)
            if math.hypot(y, x) <= kernel_radius
        }
        assert offsets == expected, (
            "The kernel must hold exactly the cells whose centre lies inside the disc. "
            f"{point_size=} {kernel_radius=} {sorted(offsets - expected)=} "
            f"{sorted(expected - offsets)=}"
        )


def test_create_circular_kernel_offsets_dilates_a_point_into_a_centred_disc() -> None:
    """Test that rendering one point at each point size covers a disc of pixels centred on the point's own pixel."""
    expected_covered_counts: Dict[float, int] = {
        1.0: 1,
        1.5: 1,
        2.0: 5,
        3.0: 9,
        4.0: 13,
        5.0: 21,
    }

    pc_data = PointCloud(xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32))
    camera = _build_camera(focal=100.0, principal_point=20.5)
    resolution = (41, 41)

    # A point size of one pixel is not dilated at all, so the single pixel it
    # covers is the point's own pixel that every wider disc must centre on.
    _, own_pixel_mask = render_depth_from_point_cloud(
        pc=pc_data,
        camera=camera,
        resolution=resolution,
        return_mask=True,
        point_size=1.0,
    )
    own_pixels = own_pixel_mask.nonzero()
    assert own_pixels.shape == (1, 2), (
        "One undilated point must cover exactly one pixel, which anchors the disc centre. "
        f"{own_pixels.shape=} {own_pixels=}"
    )
    center_y, center_x = int(own_pixels[0, 0]), int(own_pixels[0, 1])

    for point_size in POINT_SIZES:
        _, valid_mask = render_depth_from_point_cloud(
            pc=pc_data,
            camera=camera,
            resolution=resolution,
            return_mask=True,
            point_size=point_size,
        )

        covered = {
            (int(pixel[0]) - center_y, int(pixel[1]) - center_x)
            for pixel in valid_mask.nonzero()
        }
        unmatched = {(y, x) for y, x in covered if (-y, -x) not in covered}
        assert not unmatched, (
            "The covered pixels must be symmetric about the point's own pixel. "
            f"{point_size=} {center_y=} {center_x=} {sorted(unmatched)=} {sorted(covered)=}"
        )
        assert len(covered) == expected_covered_counts[point_size], (
            "A point must dilate into the disc of pixels its point size reaches. "
            f"{point_size=} {len(covered)=} "
            f"expected_count={expected_covered_counts[point_size]} {sorted(covered)=}"
        )


def _to_offset_set(kernel_offsets: torch.Tensor) -> Set[Tuple[int, int]]:
    """Collect the kernel offsets into a set of (y, x) integer pairs.

    Args:
        kernel_offsets: [num_offsets, 2] integer torch.Tensor of (y, x) offsets.

    Returns:
        The set of (y, x) int pairs the tensor carries, one per row.
    """
    offsets = {(int(offset[0]), int(offset[1])) for offset in kernel_offsets}
    assert len(offsets) == kernel_offsets.shape[0], (
        "The kernel must not repeat an offset, otherwise a disc pixel is dilated twice. "
        f"{len(offsets)=} {kernel_offsets.shape=}"
    )
    return offsets


def _build_camera(focal: float, principal_point: float) -> Camera:
    """Build an identity-pose OpenGL pinhole camera on the CPU.

    Args:
        focal: Shared focal length used for both fx and fy.
        principal_point: Shared principal-point coordinate used for both cx and cy.

    Returns:
        A Camera whose pinhole intrinsics are (fx, fy, cx, cy) and whose extrinsics are the identity cam2world matrix in the opengl convention.
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
