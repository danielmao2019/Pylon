"""TypeScript backend camera-state conversion."""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from data.structures.three_d.camera import Camera
from data.viewer.utils.controls.camera.camera_state.ts.backend.schemas.camera_state import (
    CameraState,
)


def create_camera_state_from_camera(camera: Camera) -> CameraState:
    """Convert a pinhole Camera to its three.js-native CameraState.

    Produces the ``three_trackball`` CameraState that the frontend spatial-display
    appliers consume: a ``perspective-three`` intrinsics record (aspect, near, far,
    vertical fov in degrees) and an extrinsics record carrying the camera's world
    position, camera-to-world quaternion, physical up axis, and a look-at target
    placed on the camera's forward view ray so the first paint reproduces the
    source camera's orientation.

    Args:
        camera: Camera whose extrinsics use its declared convention; its physical
            right / up / forward axes are unit vectors expressed in world
            coordinates, and ``forward`` is the direction it looks into the scene.

    Returns:
        A CameraState with ``convention == "three_trackball"``.
    """
    assert isinstance(camera, Camera), (
        "Expected camera to be a Camera. " f"{type(camera)=}"
    )

    center = camera.extrinsics.center.detach().cpu().numpy()
    right = camera.extrinsics.right.detach().cpu().numpy()
    up_axis = camera.extrinsics.up.detach().cpu().numpy()
    forward = camera.extrinsics.forward.detach().cpu().numpy()
    vertical_fov = (
        2.0 * math.atan(camera.intrinsics.cy / camera.intrinsics.fy) * 180.0 / math.pi
    )

    # three.js cameras look down local -Z, so the camera-to-world rotation has
    # basis columns [right, up, -forward]; ``as_quat`` returns (x, y, z, w).
    quaternion = Rotation.from_matrix(
        np.column_stack([right, up_axis, -forward])
    ).as_quat()

    # Pivot / look-at target on the forward view ray at content depth ||center||
    # (the exported meshes live near the world origin). Alternative on-ray depth:
    # dot(-center, forward) when positive, the exact projection of the origin.
    target = center + forward * float(np.linalg.norm(center))

    return CameraState(
        intrinsics={
            "aspect": camera.intrinsics.cx / camera.intrinsics.cy,
            "far": 1000.0,
            "fov": vertical_fov,
            "near": 0.01,
            "projection": "perspective-three",
        },
        extrinsics={
            "position": {
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
            },
            "quaternion": {
                "x": float(quaternion[0]),
                "y": float(quaternion[1]),
                "z": float(quaternion[2]),
                "w": float(quaternion[3]),
            },
            "target": {
                "x": float(target[0]),
                "y": float(target[1]),
                "z": float(target[2]),
            },
            "up": {
                "x": float(up_axis[0]),
                "y": float(up_axis[1]),
                "z": float(up_axis[2]),
            },
        },
        convention="three_trackball",
        name=camera.name,
        id=None if camera.id is None else str(camera.id),
    )
