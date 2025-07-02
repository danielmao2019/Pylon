from typing import List, Union, Tuple
import numpy as np
import torch
from scipy.spatial import cKDTree


def _normalize_points(points: torch.Tensor) -> torch.Tensor:
    """Normalize points to unbatched format (N, 3) while preserving type.
    
    Args:
        points: Input points, either [N, 3] or [1, N, 3]
        
    Returns:
        Normalized points with shape (N, 3), same type as input
    """
    if points.ndim == 2:
        # Points are unbatched [N, 3]
        assert points.shape[1] == 3, f"Points must have 3 coordinates, got shape {points.shape}"
        return points
    elif points.ndim == 3:
        # Points are batched [B, N, 3]
        assert points.shape[0] == 1, f"Batch size must be 1, got shape {points.shape}"
        assert points.shape[2] == 3, f"Points must have 3 coordinates, got shape {points.shape}"
        
        # Squeeze batch dimension
        return points.squeeze(0)
    else:
        raise ValueError(f"Points must have 2 or 3 dimensions, got shape {points.shape}")


def pc_symmetric_difference(
    src_pc: torch.Tensor,
    tgt_pc: torch.Tensor,
    radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the indices of points in the symmetric difference between two point clouds using KDTree.

    Args:
        src_pc: torch.Tensor of shape (M, 3) or (1, M, 3) - source point cloud coordinates
        tgt_pc: torch.Tensor of shape (N, 3) or (1, N, 3) - target point cloud coordinates
        radius: float - radius for neighborhood search

    Returns:
        Tuple of (src_indices, tgt_indices) - indices of points in the symmetric difference
    """
    # Input validation
    assert isinstance(src_pc, torch.Tensor), "src_pc must be a torch.Tensor"
    assert isinstance(tgt_pc, torch.Tensor), "tgt_pc must be a torch.Tensor"
    assert isinstance(radius, (int, float)), "radius must be a numeric value"
    assert radius > 0, "radius must be positive"

    # Normalize points to unbatched format
    src_pc_normalized = _normalize_points(src_pc)
    tgt_pc_normalized = _normalize_points(tgt_pc)

    # Check for empty point clouds
    if src_pc_normalized.shape[0] == 0 or tgt_pc_normalized.shape[0] == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    # Check for NaN or Inf values
    if torch.isnan(src_pc_normalized).any() or torch.isnan(tgt_pc_normalized).any() or torch.isinf(src_pc_normalized).any() or torch.isinf(tgt_pc_normalized).any():
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    # Convert to numpy for scipy's cKDTree
    src_np = src_pc_normalized.cpu().numpy()
    tgt_np = tgt_pc_normalized.cpu().numpy()

    # Build KDTree for target point cloud
    tgt_tree = cKDTree(tgt_np)

    # Find points in src_pc that are not close to any point in tgt_pc
    # Query the KDTree for all points in src_pc
    distances, _ = tgt_tree.query(src_np, k=1)  # k=1 to find the nearest neighbor
    src_diff_mask = distances > radius
    src_indices = torch.where(torch.from_numpy(src_diff_mask))[0]

    # Build KDTree for source point cloud
    src_tree = cKDTree(src_np)

    # Find points in tgt_pc that are not close to any point in src_pc
    distances, _ = src_tree.query(tgt_np, k=1)  # k=1 to find the nearest neighbor
    tgt_diff_mask = distances > radius
    tgt_indices = torch.where(torch.from_numpy(tgt_diff_mask))[0]

    return src_indices, tgt_indices
