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


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle representation to rotation matrix using Rodrigues' formula.
    
    Based on Rodrigues' rotation formula:
    R = I + sin(¸)K + (1 - cos(¸))K²
    
    Where K is the skew-symmetric cross-product matrix of the axis vector.
    
    Args:
        axis: Unit vector representing rotation axis, shape (..., 3)
        angle: Rotation angle in radians [0, 2À), shape (...) or (..., 1)
    
    Returns:
        Rotation matrix, shape (..., 3, 3)
    """
    # Ensure angle has correct shape
    if angle.dim() == axis.dim() - 1:
        angle = angle.unsqueeze(-1)
    
    # Normalize axis to ensure unit vector
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-10)
    
    # Extract axis components
    nx = axis[..., 0:1]
    ny = axis[..., 1:2]
    nz = axis[..., 2:3]
    
    # Compute trigonometric values
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    one_minus_cos = 1 - cos_theta
    
    # Build rotation matrix using Rodrigues' formula expanded form:
    # R = I*cos(¸) + nn^T*(1-cos(¸)) + K*sin(¸)
    
    # Identity scaled by cos(¸)
    batch_shape = axis.shape[:-1]
    I = torch.eye(3, device=axis.device, dtype=axis.dtype)
    I = I.expand(*batch_shape, 3, 3).clone()
    R = I * cos_theta.unsqueeze(-1)
    
    # Add outer product term: nn^T * (1-cos(¸))
    outer = torch.zeros(*batch_shape, 3, 3, device=axis.device, dtype=axis.dtype)
    outer[..., 0, 0] = nx.squeeze(-1) * nx.squeeze(-1) * one_minus_cos.squeeze(-1)
    outer[..., 0, 1] = nx.squeeze(-1) * ny.squeeze(-1) * one_minus_cos.squeeze(-1)
    outer[..., 0, 2] = nx.squeeze(-1) * nz.squeeze(-1) * one_minus_cos.squeeze(-1)
    outer[..., 1, 0] = ny.squeeze(-1) * nx.squeeze(-1) * one_minus_cos.squeeze(-1)
    outer[..., 1, 1] = ny.squeeze(-1) * ny.squeeze(-1) * one_minus_cos.squeeze(-1)
    outer[..., 1, 2] = ny.squeeze(-1) * nz.squeeze(-1) * one_minus_cos.squeeze(-1)
    outer[..., 2, 0] = nz.squeeze(-1) * nx.squeeze(-1) * one_minus_cos.squeeze(-1)
    outer[..., 2, 1] = nz.squeeze(-1) * ny.squeeze(-1) * one_minus_cos.squeeze(-1)
    outer[..., 2, 2] = nz.squeeze(-1) * nz.squeeze(-1) * one_minus_cos.squeeze(-1)
    
    R = R + outer
    
    # Add skew-symmetric term: K * sin(¸)
    K = torch.zeros(*batch_shape, 3, 3, device=axis.device, dtype=axis.dtype)
    K[..., 0, 1] = -nz.squeeze(-1) * sin_theta.squeeze(-1)
    K[..., 0, 2] = ny.squeeze(-1) * sin_theta.squeeze(-1)
    K[..., 1, 0] = nz.squeeze(-1) * sin_theta.squeeze(-1)
    K[..., 1, 2] = -nx.squeeze(-1) * sin_theta.squeeze(-1)
    K[..., 2, 0] = -ny.squeeze(-1) * sin_theta.squeeze(-1)
    K[..., 2, 1] = nx.squeeze(-1) * sin_theta.squeeze(-1)
    
    R = R + K
    
    return R


def euler_to_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to rotation matrix using XYZ convention.
    
    Composes three elementary rotations: R = Rx(±) * Ry(²) * Rz(³)
    Note: Rotations are applied in reverse order (Z first, then Y, then X).
    
    Args:
        euler_angles: Three angles [±, ², ³] in radians [0, 2À), shape (..., 3)
                     ±: rotation around X axis
                     ²: rotation around Y axis  
                     ³: rotation around Z axis
    
    Returns:
        Rotation matrix, shape (..., 3, 3)
    """
    # Extract individual angles
    alpha = euler_angles[..., 0]  # X rotation
    beta = euler_angles[..., 1]   # Y rotation
    gamma = euler_angles[..., 2]  # Z rotation
    
    # Compute trigonometric values
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)
    
    batch_shape = euler_angles.shape[:-1]
    device = euler_angles.device
    dtype = euler_angles.dtype
    
    # Build rotation matrix for XYZ convention
    # R = Rx(±) * Ry(²) * Rz(³)
    R = torch.zeros(*batch_shape, 3, 3, device=device, dtype=dtype)
    R[..., 0, 0] = cb * cg
    R[..., 0, 1] = -cb * sg
    R[..., 0, 2] = sb
    R[..., 1, 0] = sa * sb * cg + ca * sg
    R[..., 1, 1] = -sa * sb * sg + ca * cg
    R[..., 1, 2] = -sa * cb
    R[..., 2, 0] = -ca * sb * cg + sa * sg
    R[..., 2, 1] = ca * sb * sg + sa * cg
    R[..., 2, 2] = ca * cb
    
    return R


def matrix_to_axis_angle(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract axis-angle representation from rotation matrix.
    
    Uses the trace to extract angle: ¸ = arccos((trace(R) - 1) / 2)
    Axis extracted from skew-symmetric part for general case (0 < ¸ < À).
    The returned angle is normalized to [0, 2À) range.
    
    Args:
        matrix: Rotation matrix, shape (..., 3, 3)
    
    Returns:
        Tuple of (axis, angle) where:
        - axis: Unit vector representing rotation axis, shape (..., 3)
        - angle: Rotation angle in radians [0, 2À), shape (...)
    """
    # Extract rotation angle from trace
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, min=-1.0, max=1.0))
    
    batch_shape = matrix.shape[:-2]
    device = matrix.device
    dtype = matrix.dtype
    
    # Initialize axis
    axis = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
    
    # Handle different cases based on angle
    eps = 1e-6
    
    # Case 1: ¸ H 0 (identity rotation)
    identity_mask = angle < eps
    axis[identity_mask] = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    
    # Case 2: ¸ H À (180° rotation)
    pi_mask = (angle > math.pi - eps)
    if pi_mask.any():
        # Find largest diagonal element to determine dominant axis
        diag = torch.stack([matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]], dim=-1)
        max_indices = torch.argmax(diag[pi_mask], dim=-1)
        
        for idx in range(3):
            mask_idx = pi_mask.nonzero()[max_indices == idx]
            if mask_idx.numel() > 0:
                # Extract axis from symmetric part
                if idx == 0:  # Largest is R[0,0]
                    axis[mask_idx, 0] = torch.sqrt((matrix[mask_idx, 0, 0] + 1) / 2)
                    axis[mask_idx, 1] = matrix[mask_idx, 0, 1] / (2 * axis[mask_idx, 0] + eps)
                    axis[mask_idx, 2] = matrix[mask_idx, 0, 2] / (2 * axis[mask_idx, 0] + eps)
                elif idx == 1:  # Largest is R[1,1]
                    axis[mask_idx, 1] = torch.sqrt((matrix[mask_idx, 1, 1] + 1) / 2)
                    axis[mask_idx, 0] = matrix[mask_idx, 1, 0] / (2 * axis[mask_idx, 1] + eps)
                    axis[mask_idx, 2] = matrix[mask_idx, 1, 2] / (2 * axis[mask_idx, 1] + eps)
                else:  # Largest is R[2,2]
                    axis[mask_idx, 2] = torch.sqrt((matrix[mask_idx, 2, 2] + 1) / 2)
                    axis[mask_idx, 0] = matrix[mask_idx, 2, 0] / (2 * axis[mask_idx, 2] + eps)
                    axis[mask_idx, 1] = matrix[mask_idx, 2, 1] / (2 * axis[mask_idx, 2] + eps)
    
    # Case 3: General case (0 < ¸ < À)
    general_mask = ~identity_mask & ~pi_mask
    if general_mask.any():
        sin_angle = torch.sin(angle[general_mask])
        # Extract from skew-symmetric part: n = 1/(2sin(¸)) * [R32-R23, R13-R31, R21-R12]
        axis[general_mask, 0] = (matrix[general_mask, 2, 1] - matrix[general_mask, 1, 2]) / (2 * sin_angle + eps)
        axis[general_mask, 1] = (matrix[general_mask, 0, 2] - matrix[general_mask, 2, 0]) / (2 * sin_angle + eps)
        axis[general_mask, 2] = (matrix[general_mask, 1, 0] - matrix[general_mask, 0, 1]) / (2 * sin_angle + eps)
    
    # Normalize axis
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + eps)
    
    # Ensure angle is in [0, 2À) range
    angle = angle % (2 * math.pi)
    
    return axis, angle


def matrix_to_euler(matrix: torch.Tensor) -> torch.Tensor:
    """Extract Euler angles from rotation matrix using XYZ convention.
    
    For XYZ convention, extracts angles using:
    - ± (X rotation): atan2(-R23, R33)
    - ² (Y rotation): atan2(R13, sqrt(R11² + R12²))
    - ³ (Z rotation): atan2(-R12, R11)
    
    Handles gimbal lock by setting one angle to 0 when necessary.
    The returned angles are normalized to [0, 2À) range.
    
    Args:
        matrix: Rotation matrix, shape (..., 3, 3)
    
    Returns:
        Euler angles [±, ², ³] in radians [0, 2À), shape (..., 3)
        ±: rotation around X axis
        ²: rotation around Y axis
        ³: rotation around Z axis
    """
    batch_shape = matrix.shape[:-2]
    device = matrix.device
    dtype = matrix.dtype
    eps = 1e-6
    
    # Initialize output
    euler_angles = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
    
    # Check for gimbal lock
    sy = torch.sqrt(matrix[..., 0, 0]**2 + matrix[..., 1, 0]**2)
    
    # Normal case (no gimbal lock)
    normal_mask = sy > eps
    euler_angles[normal_mask, 0] = torch.atan2(-matrix[normal_mask, 1, 2], matrix[normal_mask, 2, 2])  # ± (X)
    euler_angles[normal_mask, 1] = torch.atan2(matrix[normal_mask, 0, 2], sy[normal_mask])  # ² (Y)
    euler_angles[normal_mask, 2] = torch.atan2(-matrix[normal_mask, 0, 1], matrix[normal_mask, 0, 0])  # ³ (Z)
    
    # Gimbal lock case (² H ±90°)
    gimbal_mask = ~normal_mask
    euler_angles[gimbal_mask, 0] = torch.atan2(matrix[gimbal_mask, 2, 1], matrix[gimbal_mask, 1, 1])  # ± (X)
    euler_angles[gimbal_mask, 1] = torch.atan2(matrix[gimbal_mask, 0, 2], sy[gimbal_mask])  # ² (Y)
    euler_angles[gimbal_mask, 2] = 0  # Set ³ (Z) = 0
    
    # Normalize angles to [0, 2À) range
    euler_angles = euler_angles % (2 * math.pi)
    
    return euler_angles