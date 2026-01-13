"""Pitch/yaw rotation utilities.

This module provides helper functions for converting between pitch/yaw angles
and rotation matrices while assuming zero roll. These helpers mirror the API
style of the other rotation utilities and internally reuse the Euler
conversions to keep behaviour consistent across the codebase.
"""

import torch

from data.structures.three_d.camera.camera import Camera
from utils.three_d.rotation.euler import euler_to_matrix


def pitch_yaw_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Convert [pitch, yaw] angles to a camera-to-world rotation matrix.

    Args:
        rotation: Tensor of shape (2,) containing [pitch, yaw] in radians.

    Returns:
        Rotation matrix of shape (3, 3) assuming zero roll.
    """

    rotation = rotation.reshape(-1)
    if rotation.numel() != 2:
        raise ValueError(
            f"Expected rotation to contain exactly two elements [pitch, yaw]; got {rotation.shape}"
        )

    dtype = rotation.dtype
    device = rotation.device
    pitch, yaw = rotation[0], rotation[1]

    zeros = torch.zeros(3, dtype=dtype, device=device)

    pitch_angles = zeros.clone()
    pitch_angles[0] = pitch
    yaw_angles = zeros.clone()
    yaw_angles[2] = yaw

    pitch_matrix = euler_to_matrix(pitch_angles)
    yaw_matrix = euler_to_matrix(yaw_angles)

    return yaw_matrix @ pitch_matrix


def matrix_to_pitch_yaw(rotation: torch.Tensor) -> torch.Tensor:
    """Extract [pitch, yaw] from a rotation matrix assuming zero roll.

    Args:
        rotation: Rotation matrix of shape (3, 3), interpreted as camera-to-
            world.

    Returns:
        Tensor ``[pitch, yaw]`` matching the input dtype/device.
    """

    Camera._validate_rotation_matrix(rotation)

    dtype = rotation.dtype
    device = rotation.device

    forward = rotation[:3, 1]
    # Allow small numerical error when checking unit length
    assert torch.isclose(
        torch.linalg.norm(forward),
        torch.tensor(1.0, dtype=rotation.dtype, device=rotation.device),
    ), "Expected forward vector to be unit length within tolerance"

    fx, fy, fz = forward[0], forward[1], forward[2]
    horiz = torch.sqrt(fx * fx + fy * fy)

    yaw = torch.atan2(-fx, fy)
    pitch = torch.atan2(fz, horiz)

    return torch.stack([pitch, yaw]).to(dtype=dtype, device=device)
