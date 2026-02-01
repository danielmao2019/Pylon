"""Utilities for removing roll from standard-format camera extrinsics."""

from typing import List, Tuple

import torch

_GLOBAL_UP = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)


def _find_local_up(extrinsics_list: List[torch.Tensor]) -> torch.Tensor:
    """Estimate the dominant up direction across a set of camera extrinsics.

    Args:
        extrinsics_list: List of 4x4 camera-to-world matrices in the standard convention.

    Returns:
        Unit vector representing the global up axis in the current reconstruction frame.
    """
    if not extrinsics_list:
        raise ValueError("Expected at least one extrinsics matrix to estimate local up")

    dtype = extrinsics_list[0].dtype
    device = extrinsics_list[0].device
    M = torch.zeros((3, 3), dtype=dtype, device=device)
    avg_up = torch.zeros(3, dtype=dtype, device=device)

    for extrinsics in extrinsics_list:
        if extrinsics.shape[-2:] != (4, 4):
            raise ValueError(
                f"Expected extrinsics shape (4, 4); got {extrinsics.shape}"
            )
        rotation = extrinsics[:3, :3]
        forward = rotation[:, 1]
        up = rotation[:, 2]
        right = torch.cross(forward, up)
        M = M + torch.outer(right, right)
        avg_up = avg_up + up

    eigvals, eigvecs = torch.linalg.eigh(M)
    min_index = torch.argmin(eigvals)
    local_up = eigvecs[:, min_index]
    local_up = local_up / local_up.norm()

    if avg_up.norm() > 0 and torch.dot(local_up, avg_up) < 0:
        local_up = -local_up

    return local_up


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm()
    x, y, z = axis
    K = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=axis.dtype,
        device=axis.device,
    )
    I = torch.eye(3, dtype=axis.dtype, device=axis.device)
    outer = axis[:, None] @ axis[None, :]
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    return cos * I + sin * K + (1.0 - cos) * outer


def _find_transform(local_up: torch.Tensor) -> torch.Tensor:
    """Compute the minimal rotation that aligns the local up with the global up."""
    if local_up.shape != (3,):
        raise ValueError(f"local_up must be shape (3,), got {local_up.shape}")

    local_up = local_up / local_up.norm()
    global_up = _GLOBAL_UP.to(dtype=local_up.dtype, device=local_up.device)

    dot = torch.clamp(torch.dot(local_up, global_up), min=-1.0, max=1.0)
    eps = torch.tensor(1e-6, dtype=local_up.dtype, device=local_up.device)

    transform = torch.eye(4, dtype=local_up.dtype, device=local_up.device)
    if torch.abs(dot - 1.0) < eps:
        return transform
    if torch.abs(dot + 1.0) < eps:
        reference = torch.tensor(
            [1.0, 0.0, 0.0], dtype=local_up.dtype, device=local_up.device
        )
        axis = torch.cross(local_up, reference)
        if axis.norm() < 1e-6:
            reference = torch.tensor(
                [0.0, 1.0, 0.0], dtype=local_up.dtype, device=local_up.device
            )
            axis = torch.cross(local_up, reference)
        axis_norm = axis.norm()
        if axis_norm < eps:
            return transform
        rotation = _axis_angle_to_matrix(axis, torch.pi)
        transform[:3, :3] = rotation
        return transform

    axis = torch.cross(local_up, global_up)
    axis_norm = axis.norm()
    if axis_norm < eps:
        return transform
    angle = torch.acos(dot)
    rotation = _axis_angle_to_matrix(axis, angle)
    transform[:3, :3] = rotation
    return transform


def zero_roll(
    extrinsics_list: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Apply a global leveling rotation to remove roll from camera extrinsics."""
    if not extrinsics_list:
        raise ValueError("Cannot zero roll an empty list of extrinsics")

    local_up = _find_local_up(extrinsics_list)
    transform = _find_transform(local_up)
    transform = transform.to(
        dtype=extrinsics_list[0].dtype, device=extrinsics_list[0].device
    )

    leveled = [transform @ extrinsics for extrinsics in extrinsics_list]
    return leveled, transform
