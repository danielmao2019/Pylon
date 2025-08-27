"""3D rotation transformation utilities.

This module provides functions for converting between different 3D rotation representations:
- Axis-angle representation (unit vector + angle)
- Euler angles (XYZ convention - sequential rotations around X, Y, Z axes)
- Rotation matrices (3x3 orthogonal matrices)

Based on mathematical foundations including Rodrigues' formula and Euler's rotation theorem.
"""

from typing import Tuple
import torch
import math


def axis_angle_to_matrix(
    axis: torch.Tensor, 
    angle: torch.Tensor
) -> torch.Tensor:
    """Convert axis-angle representation to rotation matrix using Rodrigues' formula.
    
    Args:
        axis: Unit vector of shape (3,) representing rotation axis
        angle: Scalar tensor representing rotation angle in radians [-pi, +pi]
        
    Returns:
        Rotation matrix of shape (3, 3)
    """
    # Ensure axis is normalized
    axis = axis / torch.norm(axis)
    
    # Construct skew-symmetric cross-product matrix K
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=axis.dtype, device=axis.device)
    
    # Apply Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    I = torch.eye(3, dtype=axis.dtype, device=axis.device)
    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    
    return R


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


def matrix_to_axis_angle(
    R: torch.Tensor,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract axis-angle representation from rotation matrix.
    
    Returns canonical form where angle is always non-negative [0, pi].
    The axis direction is chosen to ensure this canonical form.
    
    Args:
        R: Rotation matrix of shape (3, 3)
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (axis, angle) where:
        - axis: Unit vector of shape (3,)
        - angle: Scalar tensor in radians [0, pi] (always non-negative)
    """
    # Extract rotation angle from trace
    trace = torch.trace(R)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    
    # Handle special cases
    if torch.abs(angle) < eps:
        # No rotation (identity matrix)
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=R.dtype, device=R.device)
        angle = torch.tensor(0.0, dtype=R.dtype, device=R.device)
        
    elif torch.abs(angle - math.pi) < eps:
        # 180 degree rotation - need special handling
        # Find largest diagonal element to avoid numerical issues
        diag = torch.diag(R)
        k = torch.argmax(diag)
        
        # Extract axis from symmetric part
        axis = torch.zeros(3, dtype=R.dtype, device=R.device)
        axis[k] = torch.sqrt((R[k, k] + 1) / 2)
        
        # Determine signs from off-diagonal elements
        for i in range(3):
            if i != k:
                axis[i] = R[k, i] / (2 * axis[k])
                
        # Ensure unit vector
        axis = axis / torch.norm(axis)
        
    else:
        # General case: extract from skew-symmetric part
        axis = torch.tensor([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ], dtype=R.dtype, device=R.device) / (2 * torch.sin(angle))
        
        # Ensure unit vector
        axis = axis / torch.norm(axis)
    
    return axis, angle


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
        
        # Ensure canonical form: constrain beta to [-pi/2, +pi/2]
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
    else:
        # Gimbal lock case - lose one degree of freedom
        alpha = torch.atan2(R[2, 1], R[1, 1])   # X rotation
        beta = torch.atan2(R[0, 2], sy)         # Y rotation
        gamma = torch.tensor(0.0, dtype=R.dtype, device=R.device)  # Z rotation = 0
        
    angles = torch.tensor([alpha, beta, gamma], dtype=R.dtype, device=R.device)
    
    return angles
