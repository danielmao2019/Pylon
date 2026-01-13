"""Shared utilities for Level of Detail implementations."""
from typing import Dict, Any
import torch


def get_camera_position(camera_state: Dict[str, Any], device: torch.device = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Extract camera position from camera state.

    Args:
        camera_state: Camera state with 'eye' key containing position
        device: Target device for the tensor (default: CPU)
        dtype: Data type for the tensor (default: float32)

    Returns:
        Camera position as tensor (3,) on specified device
    """
    eye = camera_state.get('eye', {'x': 1.5, 'y': 1.5, 'z': 1.5})
    return torch.tensor([eye['x'], eye['y'], eye['z']], device=device, dtype=dtype)


def apply_point_constraints(current_points: int, target_points: int, min_points: int, max_reduction: float) -> int:
    """Apply minimum points and maximum reduction constraints.

    Args:
        current_points: Current number of points
        target_points: Desired target points
        min_points: Minimum points to preserve
        max_reduction: Maximum reduction ratio (0.0-1.0)

    Returns:
        Constrained target points
    """
    # Ensure minimum points
    target_points = max(target_points, min_points)

    # Ensure maximum reduction
    min_allowed = int(current_points * (1.0 - max_reduction))
    target_points = max(target_points, min_allowed)

    return target_points