
"""Rodrigues formula for axis-angle rotation utilities.

This module provides functions for converting between axis-angle representations
and rotation matrices using Rodrigues' rotation formula.
"""

from typing import Tuple
import torch
import math


def axis_angle_canonical(axis: torch.Tensor, angle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert axis-angle representation to canonical form with non-negative angle [0, pi].
    
    This resolves the axis-angle ambiguity by choosing the representation where
    the angle is always non-negative. If the input angle is negative, the axis 
    is flipped and the angle is made positive.
    
    Args:
        axis: Unit vector representing rotation axis, shape (3,)
        angle: Rotation angle in radians [-pi, +pi], scalar tensor
    
    Returns:
        Tuple of (canonical_axis, canonical_angle) where:
        - canonical_axis: Unit vector, shape (3,) 
        - canonical_angle: Angle in radians [0, pi], scalar tensor
    """
    if angle < 0:
        # Negative angle: flip axis and make angle positive
        canonical_axis = -axis
        canonical_angle = torch.abs(angle)
    else:
        # Non-negative angle: keep as is
        canonical_axis = axis
        canonical_angle = angle
    
    return canonical_axis, canonical_angle


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
    
    # Apply canonical form to result
    canonical_axis, canonical_angle = axis_angle_canonical(axis, angle)
    return canonical_axis, canonical_angle
