"""Tests for triangle-mesh rendering camera preparation."""

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from models.three_d.meshes.render.core import _prepare_cameras


def _build_camera() -> Camera:
    """Build a standard-frame pinhole camera on the CPU.

    Args:
        None.

    Returns:
        A Camera with standard image-plane intrinsics and standard pose-frame
        extrinsics.
    """
    return Camera(
        intrinsics=build_camera_intrinsics(
            model="pinhole",
            params={
                "fx": 180.0,
                "fy": 160.0,
                "cx": 72.0,
                "cy": 46.0,
                "h": 100,
                "w": 150,
            },
            intr_convention="standard",
            device=torch.device("cpu"),
        ),
        extrinsics=CameraExtrinsics(
            extrinsics=torch.tensor(
                [
                    [0.0, -1.0, 0.0, 0.25],
                    [1.0, 0.0, 0.0, -0.5],
                    [0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            extr_convention="standard",
            device=torch.device("cpu"),
        ),
        device=torch.device("cpu"),
    )


def test_prepare_cameras_restates_both_camera_frames_for_pytorch3d() -> None:
    """_prepare_cameras restates both camera frames for PyTorch3D.

    Args:
        None.

    Returns:
        None.
    """
    camera = _build_camera()
    device = torch.device("cpu")
    resolution = (200, 300)

    prepared = _prepare_cameras(
        camera=camera,
        resolution=resolution,
        device=device,
    )
    expected_camera = camera.to(
        device=device,
        intr_convention="pytorch3d",
        extr_convention="pytorch3d",
    ).scale_intrinsics(resolution=resolution)
    expected_intrinsics = expected_camera.intrinsics
    expected_extrinsics = expected_camera.extrinsics

    assert prepared.in_ndc(), "Expected PyTorch3D camera params to be in NDC."
    assert torch.allclose(
        prepared.focal_length,
        torch.tensor(
            [[expected_intrinsics.fx, expected_intrinsics.fy]], dtype=torch.float32
        ),
    ), (
        "Expected PyTorch3D focal lengths to match the converted intrinsics. "
        f"{prepared.focal_length=} {expected_intrinsics.fx=} {expected_intrinsics.fy=}"
    )
    assert torch.allclose(
        prepared.principal_point,
        torch.tensor(
            [[expected_intrinsics.cx, expected_intrinsics.cy]], dtype=torch.float32
        ),
    ), (
        "Expected PyTorch3D principal point to match the converted intrinsics. "
        f"{prepared.principal_point=} {expected_intrinsics.cx=} "
        f"{expected_intrinsics.cy=}"
    )
    assert torch.allclose(
        prepared.R,
        expected_extrinsics.w2c[:3, :3].transpose(0, 1).unsqueeze(0),
    ), (
        "Expected PyTorch3D rotation to match the converted world-to-camera pose. "
        f"{prepared.R=} {expected_extrinsics.w2c=}"
    )
    assert torch.allclose(prepared.T, expected_extrinsics.w2c[:3, 3].unsqueeze(0)), (
        "Expected PyTorch3D translation to match the converted world-to-camera pose. "
        f"{prepared.T=} {expected_extrinsics.w2c=}"
    )
