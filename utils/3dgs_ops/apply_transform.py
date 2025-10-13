"""Apply transformations to Gaussian Splatting models."""

import math
import torch
from typing import Optional
from nerfstudio.pipelines.base_pipeline import Pipeline


def quat_to_rotmat(quaternions):
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternions: [N, 4] tensor of quaternions [w, x, y, z]

    Returns:
        [N, 3, 3] rotation matrices
    """
    # Normalize quaternions
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)

    w, x, y, z = quaternions.unbind(-1)

    # Build rotation matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )

    return R


def rotmat_to_quat(rotation_matrices):
    """
    Convert rotation matrices to quaternions.

    Args:
        rotation_matrices: [N, 3, 3] rotation matrices

    Returns:
        [N, 4] quaternions [w, x, y, z]
    """
    # This is a robust implementation using Shepperd's method
    R = rotation_matrices
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)

    # Initialize quaternion tensor
    batch_size = R.shape[0]
    quaternions = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
        quaternions[mask1, 0] = 0.25 * s  # qw
        quaternions[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s  # qx
        quaternions[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s  # qy
        quaternions[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s  # qz

    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = (
            torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        )  # s = 4 * qx
        quaternions[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s  # qw
        quaternions[mask2, 1] = 0.25 * s  # qx
        quaternions[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s  # qy
        quaternions[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s  # qz

    # Case 3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = (
            torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        )  # s = 4 * qy
        quaternions[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s  # qw
        quaternions[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s  # qx
        quaternions[mask3, 2] = 0.25 * s  # qy
        quaternions[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s  # qz

    # Case 4: remaining cases
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = (
            torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        )  # s = 4 * qz
        quaternions[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s  # qw
        quaternions[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s  # qx
        quaternions[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s  # qy
        quaternions[mask4, 3] = 0.25 * s  # qz

    return quaternions


def apply_transform_to_gs(
    pipeline: Pipeline,
    rotation: Optional[torch.Tensor] = None,
    translation: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> None:
    """Apply scale, translation, and rotation transformations to a Gaussian Splatting model.

    Transformations are applied in order: scale, translation, rotation.
    This modifies the pipeline's model in-place.

    Args:
        pipeline: Nerfstudio Pipeline containing the Gaussian Splatting model
        rotation: Optional 3x3 rotation matrix as torch.Tensor
        translation: Optional 3D translation vector as torch.Tensor
        scale: Optional scale factor as float
    """
    assert isinstance(
        pipeline, Pipeline
    ), f"pipeline must be Pipeline, got {type(pipeline)}"

    with torch.no_grad():
        # Apply scale first
        if scale is not None:
            assert isinstance(scale, float), f"scale must be float, got {type(scale)}"
            pipeline.model.means[:] *= scale
            pipeline.model.scales[:] += math.log(scale)
            pipeline.model.opacities[:] *= 1.0 / (scale**3)

        # Apply translation second
        if translation is not None:
            assert isinstance(
                translation, torch.Tensor
            ), f"translation must be torch.Tensor, got {type(translation)}"
            assert (
                translation.numel() == 3
            ), f"translation must have 3 elements, got {translation.numel()}"
            translation = translation.view(1, 3)
            pipeline.model.means[:] += translation

        # Apply rotation last
        if rotation is not None:
            assert isinstance(
                rotation, torch.Tensor
            ), f"rotation must be torch.Tensor, got {type(rotation)}"
            assert rotation.shape == (
                3,
                3,
            ), f"rotation must be 3x3 matrix, got shape {rotation.shape}"

            pipeline.model.means[:] = (rotation @ pipeline.model.means.T).T
            current_rotmats = quat_to_rotmat(pipeline.model.quats)
            new_rotmats = rotation.unsqueeze(0) @ current_rotmats
            pipeline.model.quats[:] = rotmat_to_quat(new_rotmats)
