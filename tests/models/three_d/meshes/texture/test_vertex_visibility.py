"""Focused regressions for per-vertex visibility."""

import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.mesh.mesh import Mesh
from models.three_d.meshes.texture.extract.visibility.vertex_visibility import (
    compute_v_visibility_mask,
)


def _build_one_camera() -> Cameras:
    """Build one identity OpenCV camera for focused visibility tests.

    Args:
        None.

    Returns:
        One-camera batch on CPU.
    """

    assert torch.cuda.is_available(), (
        "Expected CUDA to be available for vertex-visibility regression tests. "
        f"{torch.cuda.is_available()=}"
    )

    return Cameras(
        intrinsics=[
            build_camera_intrinsics(
                model="pinhole",
                params={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
                device="cuda",
            )
        ],
        extrinsics=[
            CameraExtrinsics(
                extrinsics=torch.eye(4, dtype=torch.float32, device="cuda"),
                convention="opencv",
                device="cuda",
            )
        ],
        device="cuda",
    )


def test_compute_v_visibility_mask_keeps_some_front_facing_triangle_visibility() -> (
    None
):
    """Keep nonzero visibility when the only face is front-facing.

    Args:
        None.

    Returns:
        None.
    """

    verts = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    faces = torch.tensor([[0, 2, 1]], dtype=torch.long, device="cuda")

    visibility_mask = compute_v_visibility_mask(
        mesh=Mesh(verts=verts, faces=faces),
        camera=_build_one_camera(),
        image_height=2,
        image_width=2,
    )

    assert float(visibility_mask.sum()) > 0.0, (
        "Expected a front-facing triangle to contribute some visible verts "
        "to the one-view rasterized visibility mask. "
        f"{visibility_mask=}"
    )


def test_compute_v_visibility_mask_filters_back_facing_triangle_verts() -> None:
    """Drop verts whose only owning face is back-facing.

    Args:
        None.

    Returns:
        None.
    """

    verts = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device="cuda")

    visibility_mask = compute_v_visibility_mask(
        mesh=Mesh(verts=verts, faces=faces),
        camera=_build_one_camera(),
        image_height=2,
        image_width=2,
    )

    expected_visibility_mask = torch.zeros((3,), dtype=torch.float32, device="cuda")
    assert torch.equal(visibility_mask, expected_visibility_mask), (
        "Expected verts belonging only to a back-facing triangle to be "
        "removed from the visible set before vertex-color extraction. "
        f"{visibility_mask=} {expected_visibility_mask=}"
    )
