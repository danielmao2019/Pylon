from typing import List, Union, Tuple
import numpy as np
import torch
from scipy.spatial import cKDTree


def pc_symmetric_difference(
    src_pc: torch.Tensor,
    tgt_pc: torch.Tensor,
    radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the indices of points in the symmetric difference between two point clouds using KDTree.

    Args:
        src_pc: torch.Tensor of shape (M, 3) - source point cloud coordinates
        tgt_pc: torch.Tensor of shape (N, 3) - target point cloud coordinates
        radius: float - radius for neighborhood search

    Returns:
        Tuple of (src_indices, tgt_indices) - indices of points in the symmetric difference
    """
    # Input validation
    assert isinstance(src_pc, torch.Tensor), "src_pc must be a torch.Tensor"
    assert isinstance(tgt_pc, torch.Tensor), "tgt_pc must be a torch.Tensor"
    assert isinstance(radius, (int, float)), "radius must be a numeric value"
    assert radius > 0, "radius must be positive"

    # Check tensor dimensions
    assert src_pc.dim() == 2, f"src_pc must be 2D tensor, got {src_pc.dim()}D"
    assert tgt_pc.dim() == 2, f"tgt_pc must be 2D tensor, got {tgt_pc.dim()}D"
    assert src_pc.shape[1] == 3, f"src_pc must have shape (M, 3), got {src_pc.shape}"
    assert tgt_pc.shape[1] == 3, f"tgt_pc must have shape (N, 3), got {tgt_pc.shape}"

    # Check for empty point clouds
    if src_pc.shape[0] == 0 or tgt_pc.shape[0] == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    # Check for NaN or Inf values
    if torch.isnan(src_pc).any() or torch.isnan(tgt_pc).any() or torch.isinf(src_pc).any() or torch.isinf(tgt_pc).any():
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    # Convert to numpy for scipy's cKDTree
    src_np = src_pc.cpu().numpy()
    tgt_np = tgt_pc.cpu().numpy()

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
