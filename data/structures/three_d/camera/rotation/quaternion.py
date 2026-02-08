"""Quaternion conversion utilities."""

import numpy as np
import torch


def quat_to_rotmat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to rotation matrices.

    Args:
        quaternions: [N, 4] tensor of quaternions [w, x, y, z]

    Returns:
        [N, 3, 3] rotation matrices
    """
    assert isinstance(
        quaternions, torch.Tensor
    ), f"quaternions must be torch.Tensor, got {type(quaternions)}"
    assert quaternions.ndim == 2, f"quaternions must be 2D, got {quaternions.ndim}D"
    assert (
        quaternions.shape[1] == 4
    ), f"quaternions must have shape [N,4], got {quaternions.shape}"

    normalized_quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
    w, x, y, z = normalized_quaternions.unbind(-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rotation_matrices = torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )

    return rotation_matrices


def rotmat_to_quat(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions.

    Args:
        rotation_matrices: [N, 3, 3] rotation matrices

    Returns:
        [N, 4] quaternions [w, x, y, z]
    """
    assert isinstance(
        rotation_matrices, torch.Tensor
    ), f"rotation_matrices must be torch.Tensor, got {type(rotation_matrices)}"
    assert (
        rotation_matrices.ndim == 3
    ), f"rotation_matrices must be 3D, got {rotation_matrices.ndim}D"
    assert rotation_matrices.shape[1:] == (
        3,
        3,
    ), f"rotation_matrices must have shape [N,3,3], got {rotation_matrices.shape}"

    trace = rotation_matrices.diagonal(dim1=-2, dim2=-1).sum(-1)

    batch_size = rotation_matrices.shape[0]
    quaternions = torch.zeros(
        batch_size, 4, device=rotation_matrices.device, dtype=rotation_matrices.dtype
    )

    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2
        quaternions[mask1, 0] = 0.25 * s
        quaternions[mask1, 1] = (
            rotation_matrices[mask1, 2, 1] - rotation_matrices[mask1, 1, 2]
        ) / s
        quaternions[mask1, 2] = (
            rotation_matrices[mask1, 0, 2] - rotation_matrices[mask1, 2, 0]
        ) / s
        quaternions[mask1, 3] = (
            rotation_matrices[mask1, 1, 0] - rotation_matrices[mask1, 0, 1]
        ) / s

    mask2 = (
        (~mask1)
        & (rotation_matrices[:, 0, 0] > rotation_matrices[:, 1, 1])
        & (rotation_matrices[:, 0, 0] > rotation_matrices[:, 2, 2])
    )
    if mask2.any():
        s = (
            torch.sqrt(
                1.0
                + rotation_matrices[mask2, 0, 0]
                - rotation_matrices[mask2, 1, 1]
                - rotation_matrices[mask2, 2, 2]
            )
            * 2
        )
        quaternions[mask2, 0] = (
            rotation_matrices[mask2, 2, 1] - rotation_matrices[mask2, 1, 2]
        ) / s
        quaternions[mask2, 1] = 0.25 * s
        quaternions[mask2, 2] = (
            rotation_matrices[mask2, 0, 1] + rotation_matrices[mask2, 1, 0]
        ) / s
        quaternions[mask2, 3] = (
            rotation_matrices[mask2, 0, 2] + rotation_matrices[mask2, 2, 0]
        ) / s

    mask3 = (
        (~mask1) & (~mask2) & (rotation_matrices[:, 1, 1] > rotation_matrices[:, 2, 2])
    )
    if mask3.any():
        s = (
            torch.sqrt(
                1.0
                + rotation_matrices[mask3, 1, 1]
                - rotation_matrices[mask3, 0, 0]
                - rotation_matrices[mask3, 2, 2]
            )
            * 2
        )
        quaternions[mask3, 0] = (
            rotation_matrices[mask3, 0, 2] - rotation_matrices[mask3, 2, 0]
        ) / s
        quaternions[mask3, 1] = (
            rotation_matrices[mask3, 0, 1] + rotation_matrices[mask3, 1, 0]
        ) / s
        quaternions[mask3, 2] = 0.25 * s
        quaternions[mask3, 3] = (
            rotation_matrices[mask3, 1, 2] + rotation_matrices[mask3, 2, 1]
        ) / s

    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = (
            torch.sqrt(
                1.0
                + rotation_matrices[mask4, 2, 2]
                - rotation_matrices[mask4, 0, 0]
                - rotation_matrices[mask4, 1, 1]
            )
            * 2
        )
        quaternions[mask4, 0] = (
            rotation_matrices[mask4, 1, 0] - rotation_matrices[mask4, 0, 1]
        ) / s
        quaternions[mask4, 1] = (
            rotation_matrices[mask4, 0, 2] + rotation_matrices[mask4, 2, 0]
        ) / s
        quaternions[mask4, 2] = (
            rotation_matrices[mask4, 1, 2] + rotation_matrices[mask4, 2, 1]
        ) / s
        quaternions[mask4, 3] = 0.25 * s

    return quaternions


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix.

    Args:
        qvec: Quaternion [qw, qx, qy, qz]

    Returns:
        3x3 rotation matrix
    """
    assert isinstance(qvec, np.ndarray), f"qvec must be np.ndarray, got {type(qvec)}"
    assert qvec.shape == (4,), f"qvec must have shape (4,), got {qvec.shape}"
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [qw, qx, qy, qz]
    """
    assert isinstance(R, np.ndarray), f"R must be np.ndarray, got {type(R)}"
    assert R.shape == (3, 3), f"R must have shape (3,3), got {R.shape}"
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0.0, 0.0, 0.0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0.0, 0.0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0.0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ],
            dtype=R.dtype,
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
