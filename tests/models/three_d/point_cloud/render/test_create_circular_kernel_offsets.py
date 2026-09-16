"""Test cases for the circular kernel offsets that dilate a rendered point into a disc."""

import math

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


def test_create_circular_kernel_offsets_disc_is_centred() -> None:
    """The kernel reaches equally on both sides of the origin, since a disc that reaches farther one way grows every rendered point off its own pixel.

    Args:
        None.

    Returns:
        None.
    """
    for point_size in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        kernel_offsets = create_circular_kernel_offsets(
            point_size=point_size, device=torch.device("cpu")
        )
        offsets = set()
        for offset in kernel_offsets:
            offsets.add((int(offset[0]), int(offset[1])))

        # The offsets whose negation is missing from the same set.
        unmatched = set()
        for y, x in offsets:
            if (-y, -x) not in offsets:
                unmatched.add((y, x))
        assert not unmatched, (
            "Every kernel offset must have its negation in the kernel, otherwise the disc "
            "reaches farther on one side of the point than on the other. "
            f"{point_size=} {sorted(unmatched)=} {sorted(offsets)=}"
        )


def test_create_circular_kernel_offsets_membership_is_the_radius_rule() -> None:
    """The kernel is exactly the cells whose centre lies inside the disc, neither more nor fewer, checked against a radius rule the test derives itself.

    Args:
        None.

    Returns:
        None.
    """
    for point_size in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        kernel_offsets = create_circular_kernel_offsets(
            point_size=point_size, device=torch.device("cpu")
        )
        offsets = set()
        for offset in kernel_offsets:
            offsets.add((int(offset[0]), int(offset[1])))

        kernel_radius = point_size / 2.0
        # A generous search box, one cell past the disc.
        search_reach = math.ceil(point_size) + 1
        # The integer cells of the search box whose distance from the origin is within kernel_radius.
        expected = set()
        for y in range(-search_reach, search_reach + 1):
            for x in range(-search_reach, search_reach + 1):
                if math.hypot(y, x) <= kernel_radius:
                    expected.add((y, x))
        assert offsets == expected and len(offsets) == kernel_offsets.shape[0], (
            "The kernel must hold exactly the cells whose centre lies inside the disc, and must not repeat a cell, otherwise a disc pixel is dilated twice. "
            f"{point_size=} {kernel_radius=} {sorted(offsets - expected)=} "
            f"{sorted(expected - offsets)=} {len(offsets)=} {kernel_offsets.shape=}"
        )


def test_create_circular_kernel_offsets_dilates_a_point_into_a_centred_disc() -> None:
    """One rendered point grows into a disc centred on its own pixel, which is the kernel's symmetry seen through the renderer that uses it.

    Args:
        None.

    Returns:
        None.
    """
    # The pixel count each point size's disc holds.
    expected_covered_counts = {1.0: 1, 1.5: 1, 2.0: 5, 3.0: 9, 4.0: 13, 5.0: 21}
    principal_point = 20.5
    pc_data = PointCloud(xyz=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32))
    camera = _build_camera(focal=100.0, principal_point=principal_point)
    resolution = (41, 41)

    # The point sits on the optical axis, so it lands on the pixel holding the principal point, the same pixel index on both axes.
    center_pixel = math.floor(principal_point)

    for point_size in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        _, valid_mask = render_depth_from_point_cloud(
            pc=pc_data,
            camera=camera,
            resolution=resolution,
            return_mask=True,
            point_size=point_size,
        )

        # The covered pixels as offsets from the pixel the point itself landed on.
        covered = set()
        for pixel in valid_mask.nonzero():
            covered.add((int(pixel[0]) - center_pixel, int(pixel[1]) - center_pixel))

        unmatched = set()
        for y, x in covered:
            if (-y, -x) not in covered:
                unmatched.add((y, x))
        assert not unmatched, (
            "The covered pixels must be symmetric about the point's own pixel. "
            f"{point_size=} {center_pixel=} {sorted(unmatched)=} {sorted(covered)=}"
        )
        assert len(covered) == expected_covered_counts[point_size], (
            "A point must dilate into the disc of pixels its point size reaches. "
            f"{point_size=} {len(covered)=} "
            f"expected_count={expected_covered_counts[point_size]} {sorted(covered)=}"
        )


def _build_camera(focal: float, principal_point: float) -> Camera:
    """Builds the identity-pose OpenGL pinhole camera on the CPU whose own extents match the requested resolution, so no intrinsics rescaling moves the point off centre.

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
