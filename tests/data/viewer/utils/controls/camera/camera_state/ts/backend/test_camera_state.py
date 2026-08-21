"""Tests for the three.js-native backend camera-state conversion."""

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.viewer.utils.controls.camera.camera_state.ts.backend.camera_state import (
    create_camera_state_from_camera,
)
from data.viewer.utils.controls.camera.camera_state.ts.backend.schemas.camera_state import (
    CameraState,
)


def _make_camera() -> Camera:
    """Build a CPU opencv Camera fixture with a non-trivial rotation.

    Args:
        None.

    Returns:
        A Camera on the CPU with pinhole intrinsics, a rotated opencv pose, name, and id.
    """
    rotation = Rotation.from_euler(
        seq="xyz", angles=[20.0, -35.0, 15.0], degrees=True
    ).as_matrix()
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[:3, :3] = torch.tensor(rotation, dtype=torch.float32)
    matrix[:3, 3] = torch.tensor([0.3, -0.2, 1.1], dtype=torch.float32)
    extrinsics = CameraExtrinsics(extrinsics=matrix, convention="opencv", device="cpu")
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    return Camera(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        name="frame_0",
        id=7,
        device="cpu",
    )


def test_create_camera_state_produces_three_trackball_state() -> None:
    """Emit the three_trackball CameraState the frontend appliers consume.

    Args:
        None.

    Returns:
        None.
    """
    camera = _make_camera()
    state = create_camera_state_from_camera(camera)

    assert isinstance(state, CameraState), f"{type(state)=}"
    assert state.convention == "three_trackball", f"{state.convention=}"
    assert state.name == "frame_0", f"{state.name=}"
    assert state.id == "7", f"{state.id=}"

    # --- intrinsics: perspective-three record with vertical fov ---
    assert set(state.intrinsics) == {
        "aspect",
        "far",
        "fov",
        "near",
        "projection",
    }, f"{set(state.intrinsics)=}"
    assert (
        state.intrinsics["projection"] == "perspective-three"
    ), f"{state.intrinsics['projection']=}"
    assert state.intrinsics["near"] == 0.01, f"{state.intrinsics['near']=}"
    assert state.intrinsics["far"] == 1000.0, f"{state.intrinsics['far']=}"
    expected_fov = (
        2.0 * math.atan(camera.intrinsics.cy / camera.intrinsics.fy) * 180.0 / math.pi
    )
    assert (
        abs(state.intrinsics["fov"] - expected_fov) < 1e-6
    ), f"{state.intrinsics['fov']=} {expected_fov=}"
    expected_aspect = camera.intrinsics.cx / camera.intrinsics.cy
    assert (
        abs(state.intrinsics["aspect"] - expected_aspect) < 1e-6
    ), f"{state.intrinsics['aspect']=} {expected_aspect=}"

    # --- extrinsics: position / quaternion / target / up ---
    assert set(state.extrinsics) == {
        "position",
        "quaternion",
        "target",
        "up",
    }, f"{set(state.extrinsics)=}"

    center = camera.extrinsics.center.detach().cpu().numpy()
    right = camera.extrinsics.right.detach().cpu().numpy()
    up_axis = camera.extrinsics.up.detach().cpu().numpy()
    forward = camera.extrinsics.forward.detach().cpu().numpy()

    position = np.array(
        [
            state.extrinsics["position"]["x"],
            state.extrinsics["position"]["y"],
            state.extrinsics["position"]["z"],
        ]
    )
    assert np.allclose(position, center, atol=1e-5), f"{position=} {center=}"

    up = np.array(
        [
            state.extrinsics["up"]["x"],
            state.extrinsics["up"]["y"],
            state.extrinsics["up"]["z"],
        ]
    )
    assert np.allclose(up, up_axis, atol=1e-5), f"{up=} {up_axis=}"

    # target lies on the forward view ray (same direction as forward)
    target = np.array(
        [
            state.extrinsics["target"]["x"],
            state.extrinsics["target"]["y"],
            state.extrinsics["target"]["z"],
        ]
    )
    ray = target - position
    ray_direction = ray / np.linalg.norm(ray)
    assert np.allclose(
        ray_direction, forward, atol=1e-4
    ), f"{ray_direction=} {forward=}"

    # quaternion reconstructs the three.js [right, up, -forward] basis
    quaternion = np.array(
        [
            state.extrinsics["quaternion"]["x"],
            state.extrinsics["quaternion"]["y"],
            state.extrinsics["quaternion"]["z"],
            state.extrinsics["quaternion"]["w"],
        ]
    )
    reconstructed_basis = Rotation.from_quat(quaternion).as_matrix()
    expected_basis = np.column_stack([right, up_axis, -forward])
    assert np.allclose(
        reconstructed_basis, expected_basis, atol=1e-5
    ), f"{reconstructed_basis=} {expected_basis=}"
