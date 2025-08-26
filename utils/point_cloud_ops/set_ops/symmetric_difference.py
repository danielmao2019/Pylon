from typing import List, Union, Tuple
import torch
from utils.point_cloud_ops.knn.knn import knn
from utils.input_checks.point_cloud import check_pc_xyz


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

    # Validate using check_pc_xyz (handles empty, NaN, Inf checks)
    check_pc_xyz(src_pc_normalized)
    check_pc_xyz(tgt_pc_normalized)

    # Find points in src_pc that are not close to any point in tgt_pc
    # Query nearest neighbor for each source point in target cloud
    distances_src_to_tgt, _ = knn(
        query_points=src_pc_normalized,
        reference_points=tgt_pc_normalized,
        k=1,  # Find nearest neighbor
        return_distances=True
    )
    
    # Source points beyond radius are in symmetric difference
    src_diff_mask = distances_src_to_tgt.squeeze(1) > radius
    src_indices = torch.where(src_diff_mask)[0]

    # Find points in tgt_pc that are not close to any point in src_pc
    # Query nearest neighbor for each target point in source cloud
    distances_tgt_to_src, _ = knn(
        query_points=tgt_pc_normalized,
        reference_points=src_pc_normalized,
        k=1,  # Find nearest neighbor
        return_distances=True
    )
    
    # Target points beyond radius are in symmetric difference
    tgt_diff_mask = distances_tgt_to_src.squeeze(1) > radius
    tgt_indices = torch.where(tgt_diff_mask)[0]

    return src_indices, tgt_indices
