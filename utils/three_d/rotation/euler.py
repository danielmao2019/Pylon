"""Euler angles rotation utilities.

This module provides functions for converting between Euler angles and rotation matrices,
including canonical form helpers to resolve Euler angle ambiguity.
"""

import math

import torch

from data.structures.three_d.camera.validation import validate_rotation_matrix


def euler_to_matrix(angles: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to rotation matrix using XYZ convention.

    Args:
        angles: Tensor of shape (3,) containing rotation angles in radians [-pi, +pi]
                [X rotation, Y rotation, Z rotation]

    Returns:
        Rotation matrix of shape (3, 3)
    """
    # Extract individual angles for XYZ convention
    alpha = angles[0]  # X rotation
    beta = angles[1]  # Y rotation
    gamma = angles[2]  # Z rotation

    # Compute trigonometric values
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)

    # Rotation around X axis
    R_x = torch.tensor(
        [[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=angles.dtype, device=angles.device
    )

    # Rotation around Y axis
    R_y = torch.tensor(
        [[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=angles.dtype, device=angles.device
    )

    # Rotation around Z axis
    R_z = torch.tensor(
        [[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], dtype=angles.dtype, device=angles.device
    )

    # Compose rotations: R = Rx @ Ry @ Rz
    # Applied in reverse order: Z first, then Y, then X
    R = R_x @ R_y @ R_z

    return R


def matrix_to_euler(
    R: torch.Tensor, order: str = 'xyz', eps: float = 1e-6
) -> torch.Tensor:
    """Extract Euler angles from rotation matrix using specified convention.

    Returns canonical form where Y rotation is constrained to [-pi/2, +pi/2]
    to resolve Euler angle ambiguity.

    Args:
        R: Rotation matrix of shape (3, 3)
        order: Euler angle convention ('xyz', 'zxy', or 'yzx'), default 'xyz'
        eps: Small value for gimbal lock detection

    Returns:
        Tensor of shape (3,) containing Euler angles in radians [-pi, +pi]
        Order of angles depends on the convention:
        - 'xyz': [X rotation, Y rotation, Z rotation]
        - 'zxy': [Z rotation, X rotation, Y rotation]
        - 'yzx': [Y rotation, Z rotation, X rotation]
        Y rotation is constrained to [-pi/2, +pi/2] for canonical form
    """
    validate_rotation_matrix(R)

    # Validate order parameter
    assert isinstance(order, str), f"order must be a string, got {type(order)}"
    assert order in [
        'xyz',
        'zxy',
        'yzx',
    ], f"order must be 'xyz', 'zxy', or 'yzx', got '{order}'"

    # Extract angles using appropriate helper function
    if order == 'xyz':
        angles = _matrix_to_euler_xyz(R, eps)
    elif order == 'zxy':
        angles = _matrix_to_euler_zxy(R, eps)
    elif order == 'yzx':
        angles = _matrix_to_euler_yzx(R, eps)
    else:
        raise ValueError(f"Unsupported order: {order}")

    # Apply canonical normalization to constrain Y rotation to [-pi/2, +pi/2]
    return euler_canonical(angles, order=order)


def _matrix_to_euler_xyz(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Extract Euler angles from rotation matrix using XYZ convention.

    Returns canonical form where Y rotation is constrained to [-pi/2, +pi/2]
    to resolve Euler angle ambiguity.

    Args:
        R: Rotation matrix of shape (3, 3)
        eps: Small value for gimbal lock detection

    Returns:
        Tensor of shape (3,) containing Euler angles in radians [-pi, +pi]
        [X rotation, Y rotation, Z rotation]
        Y rotation is constrained to [-pi/2, +pi/2] for canonical form
    """
    # Check for gimbal lock
    sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < eps  # Gimbal lock when cos(beta) ≈ 0

    if not singular:
        # Normal case - no gimbal lock
        # For XYZ Euler convention: R = Rx(alpha) * Ry(beta) * Rz(gamma)
        # Standard extraction formulas:
        beta = torch.asin(torch.clamp(R[0, 2], -1, 1))  # Y rotation in [-pi/2, +pi/2]
        alpha = torch.atan2(-R[1, 2], R[2, 2])  # X rotation
        gamma = torch.atan2(-R[0, 1], R[0, 0])  # Z rotation

    else:
        # Gimbal lock case - lose one degree of freedom
        alpha = torch.atan2(R[2, 1], R[1, 1])  # X rotation
        beta = torch.atan2(R[0, 2], sy)  # Y rotation
        gamma = torch.tensor(0.0, dtype=R.dtype, device=R.device)  # Z rotation = 0

    angles = torch.tensor([alpha, beta, gamma], dtype=R.dtype, device=R.device)
    return angles


def _matrix_to_euler_zxy(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Extract Euler angles from rotation matrix using ZXY convention.

    Returns canonical form where X rotation is constrained to [-pi/2, +pi/2]
    to resolve Euler angle ambiguity.

    Args:
        R: Rotation matrix of shape (3, 3)
        eps: Small value for gimbal lock detection

    Returns:
        Tensor of shape (3,) containing Euler angles in radians [-pi, +pi]
        [Z rotation, X rotation, Y rotation]
        X rotation is constrained to [-pi/2, +pi/2] for canonical form
    """
    # For ZXY Euler convention: R = Rz(alpha) * Rx(beta) * Ry(gamma)
    # Matrix element R[2,1] = sin(beta)
    # Check for gimbal lock when cos(beta) ≈ 0
    cos_beta = torch.sqrt(1.0 - torch.clamp(R[2, 1] ** 2, 0, 1))
    singular = cos_beta < eps

    if not singular:
        # Normal case - no gimbal lock
        beta = torch.asin(torch.clamp(R[2, 1], -1, 1))  # X rotation in [-pi/2, +pi/2]
        alpha = torch.atan2(-R[0, 1], R[1, 1])  # Z rotation
        gamma = torch.atan2(-R[2, 0], R[2, 2])  # Y rotation

    else:
        # Gimbal lock case - lose one degree of freedom
        beta = torch.asin(torch.clamp(R[2, 1], -1, 1))  # X rotation
        alpha = torch.atan2(R[0, 2], R[0, 0])  # Z rotation
        gamma = torch.tensor(0.0, dtype=R.dtype, device=R.device)  # Y rotation = 0

    angles = torch.tensor([alpha, beta, gamma], dtype=R.dtype, device=R.device)
    return angles


def _matrix_to_euler_yzx(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Extract Euler angles from rotation matrix using YZX convention.

    Returns canonical form where Z rotation is constrained to [-pi/2, +pi/2]
    to resolve Euler angle ambiguity.

    Args:
        R: Rotation matrix of shape (3, 3)
        eps: Small value for gimbal lock detection

    Returns:
        Tensor of shape (3,) containing Euler angles in radians [-pi, +pi]
        [Y rotation, Z rotation, X rotation]
        Z rotation is constrained to [-pi/2, +pi/2] for canonical form
    """
    # For YZX Euler convention: R = Ry(alpha) * Rz(beta) * Rx(gamma)
    # Matrix element R[1,0] = sin(beta)
    # Check for gimbal lock when cos(beta) ≈ 0
    cos_beta = torch.sqrt(1.0 - torch.clamp(R[1, 0] ** 2, 0, 1))
    singular = cos_beta < eps

    if not singular:
        # Normal case - no gimbal lock
        beta = torch.asin(torch.clamp(R[1, 0], -1, 1))  # Z rotation in [-pi/2, +pi/2]
        alpha = torch.atan2(-R[2, 0], R[0, 0])  # Y rotation
        gamma = torch.atan2(-R[1, 2], R[1, 1])  # X rotation

    else:
        # Gimbal lock case - lose one degree of freedom
        beta = torch.asin(torch.clamp(R[1, 0], -1, 1))  # Z rotation
        alpha = torch.atan2(R[2, 1], R[2, 2])  # Y rotation
        gamma = torch.tensor(0.0, dtype=R.dtype, device=R.device)  # X rotation = 0

    angles = torch.tensor([alpha, beta, gamma], dtype=R.dtype, device=R.device)
    return angles


def euler_canonical(angles: torch.Tensor, order: str = 'xyz') -> torch.Tensor:
    """Convert Euler angles to canonical form where Y rotation is in [-pi/2, +pi/2].

    This resolves the Euler angle ambiguity by choosing the representation where
    the Y rotation is constrained to [-pi/2, +pi/2].

    Args:
        angles: Euler angles in radians [-pi, +pi], shape (3,)
                Order depends on convention:
                - 'xyz': [X rotation, Y rotation, Z rotation]
                - 'zxy': [Z rotation, X rotation, Y rotation]
                - 'yzx': [Y rotation, Z rotation, X rotation]
        order: Euler angle convention ('xyz', 'zxy', or 'yzx'), default 'xyz'

    Returns:
        Canonical Euler angles with Y rotation constrained to [-pi/2, +pi/2], shape (3,)
    """
    # Validate order parameter
    assert isinstance(order, str), f"order must be a string, got {type(order)}"
    assert order in [
        'xyz',
        'zxy',
        'yzx',
    ], f"order must be 'xyz', 'zxy', or 'yzx', got '{order}'"

    alpha, beta, gamma = angles[0], angles[1], angles[2]

    # Find Y rotation index based on order
    if order == 'xyz':
        y_index = 1  # Y is at index 1
    elif order == 'zxy':
        y_index = 2  # Y is at index 2
    elif order == 'yzx':
        y_index = 0  # Y is at index 0
    else:
        raise ValueError(f"Unsupported order: {order}")

    # Get Y rotation value
    if y_index == 0:
        y_rot = alpha
    elif y_index == 1:
        y_rot = beta
    else:
        y_rot = gamma

    # Constrain Y rotation to [-pi/2, +pi/2]
    if y_rot > math.pi / 2:
        # Y > pi/2: use alternate solution (alpha+pi, pi-Y, gamma+pi)
        alpha = alpha + math.pi if alpha <= 0 else alpha - math.pi
        if y_index == 0:
            alpha = math.pi - alpha
        elif y_index == 1:
            beta = math.pi - beta
        else:
            gamma = math.pi - gamma
        gamma = gamma + math.pi if gamma <= 0 else gamma - math.pi
    elif y_rot < -math.pi / 2:
        # Y < -pi/2: use alternate solution (alpha+pi, -pi-Y, gamma+pi)
        alpha = alpha + math.pi if alpha <= 0 else alpha - math.pi
        if y_index == 0:
            alpha = -math.pi - alpha
        elif y_index == 1:
            beta = -math.pi - beta
        else:
            gamma = -math.pi - gamma
        gamma = gamma + math.pi if gamma <= 0 else gamma - math.pi

    return torch.tensor([alpha, beta, gamma], dtype=angles.dtype, device=angles.device)
