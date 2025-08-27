
"""Euler angles rotation utilities.

This module provides functions for converting between Euler angles and rotation matrices,
including canonical form helpers to resolve Euler angle ambiguity.
"""

import torch
import math


def euler_canonical(angles: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to canonical form where Y rotation is in [-pi/2, +pi/2].
    
    This resolves the Euler angle ambiguity by choosing the representation where
    the middle rotation (Y axis, beta) is constrained to [-pi/2, +pi/2].
    
    Args:
        angles: Euler angles [alpha, beta, gamma] in radians [-pi, +pi], shape (3,)
    
    Returns:
        Canonical Euler angles with beta constrained to [-pi/2, +pi/2], shape (3,)
    """
    alpha, beta, gamma = angles[0], angles[1], angles[2]
    
    # Constrain beta to [-pi/2, +pi/2]
    if beta > math.pi / 2:
        # beta > pi/2: use alternate solution (alpha+pi, pi-beta, gamma+pi)
        alpha = alpha + math.pi if alpha <= 0 else alpha - math.pi
        beta = math.pi - beta
        gamma = gamma + math.pi if gamma <= 0 else gamma - math.pi
    elif beta < -math.pi / 2:
        # beta < -pi/2: use alternate solution (alpha+pi, -pi-beta, gamma+pi)  
        alpha = alpha + math.pi if alpha <= 0 else alpha - math.pi
        beta = -math.pi - beta
        gamma = gamma + math.pi if gamma <= 0 else gamma - math.pi
        
    return torch.tensor([alpha, beta, gamma], dtype=angles.dtype, device=angles.device)


def euler_to_matrix(
    angles: torch.Tensor
) -> torch.Tensor:
    """Convert Euler angles to rotation matrix using XYZ convention.
    
    Args:
        angles: Tensor of shape (3,) containing rotation angles in radians [-pi, +pi]
                [X rotation, Y rotation, Z rotation]
        
    Returns:
        Rotation matrix of shape (3, 3)
    """
    # Extract individual angles for XYZ convention
    alpha = angles[0]  # X rotation
    beta = angles[1]   # Y rotation
    gamma = angles[2]  # Z rotation
    
    # Compute trigonometric values
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)
    
    # Rotation around X axis
    R_x = torch.tensor([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa, ca]
    ], dtype=angles.dtype, device=angles.device)
    
    # Rotation around Y axis
    R_y = torch.tensor([
        [cb, 0, sb],
        [0, 1, 0],
        [-sb, 0, cb]
    ], dtype=angles.dtype, device=angles.device)
    
    # Rotation around Z axis
    R_z = torch.tensor([
        [cg, -sg, 0],
        [sg, cg, 0],
        [0, 0, 1]
    ], dtype=angles.dtype, device=angles.device)
    
    # Compose rotations: R = Rx @ Ry @ Rz
    # Applied in reverse order: Z first, then Y, then X
    R = R_x @ R_y @ R_z
    
    return R




def matrix_to_euler(
    R: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
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
    sy = torch.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < eps  # Gimbal lock when cos(beta) ≈ 0
    
    if not singular:
        # Normal case - no gimbal lock
        alpha = torch.atan2(-R[1, 2], R[2, 2])  # X rotation
        beta = torch.atan2(R[0, 2], sy)         # Y rotation
        gamma = torch.atan2(-R[0, 1], R[0, 0])  # Z rotation
        
        # Apply canonical form constraint directly during extraction
        # This ensures beta is in [-pi/2, +pi/2] for canonical form
    else:
        # Gimbal lock case - lose one degree of freedom
        alpha = torch.atan2(R[2, 1], R[1, 1])   # X rotation
        beta = torch.atan2(R[0, 2], sy)         # Y rotation
        gamma = torch.tensor(0.0, dtype=R.dtype, device=R.device)  # Z rotation = 0
        
    angles = torch.tensor([alpha, beta, gamma], dtype=R.dtype, device=R.device)
    
    # Apply canonical form to result
    canonical_angles = euler_canonical(angles)
    return canonical_angles
