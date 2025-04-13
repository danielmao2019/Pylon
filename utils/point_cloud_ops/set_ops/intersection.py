from typing import Tuple
import torch
from scipy.spatial import cKDTree


def pc_intersection(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the intersection between two point clouds.
    
    Args:
        src_points: Source point cloud positions, shape (N, 3)
        tgt_points: Target point cloud positions, shape (M, 3)
        radius: Distance radius for considering points as overlapping
        
    Returns:
        A tuple containing:
        - Indices of source points that are close to any target point
        - Indices of target points that are close to any source point
    """
    assert isinstance(src_points, torch.Tensor)
    assert isinstance(tgt_points, torch.Tensor)
    assert src_points.ndim == 2 and tgt_points.ndim == 2
    assert src_points.shape[1] == 3 and tgt_points.shape[1] == 3
    
    # Convert to numpy for KD-tree operations
    src_np = src_points.cpu().numpy()
    tgt_np = tgt_points.cpu().numpy()
    
    # Build KD-tree for target points
    tgt_tree = cKDTree(tgt_np)
    
    # Find source points that are close to any target point
    # Query with radius returns all points within radius
    src_overlapping_indices = []
    for i, src_point in enumerate(src_np):
        # Find all target points within radius of this source point
        neighbors = tgt_tree.query_ball_point(src_point, radius)
        if len(neighbors) > 0:
            src_overlapping_indices.append(i)
    
    # Build KD-tree for source points
    src_tree = cKDTree(src_np)
    
    # Find target points that are close to any source point
    tgt_overlapping_indices = []
    for i, tgt_point in enumerate(tgt_np):
        # Find all source points within radius of this target point
        neighbors = src_tree.query_ball_point(tgt_point, radius)
        if len(neighbors) > 0:
            tgt_overlapping_indices.append(i)
    
    # Convert lists to tensors
    src_overlapping_indices = torch.tensor(src_overlapping_indices, device=src_points.device)
    tgt_overlapping_indices = torch.tensor(tgt_overlapping_indices, device=tgt_points.device)
    
    return src_overlapping_indices, tgt_overlapping_indices


def compute_pc_iou(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float,
) -> float:
    """
    Calculate the overlap ratio (IoU) between two point clouds.
    
    Args:
        src_points: Source point cloud positions, shape (N, 3)
        tgt_points: Target point cloud positions, shape (M, 3)
        radius: Distance radius for considering points as overlapping
        
    Returns:
        The overlap ratio, defined as the number of overlapping points divided by the total number of points
    """
    assert isinstance(src_points, torch.Tensor)
    assert isinstance(tgt_points, torch.Tensor)
    assert src_points.ndim == 2 and tgt_points.ndim == 2
    assert src_points.shape[1] == 3 and tgt_points.shape[1] == 3
    # Get overlapping indices
    src_overlapping_indices, tgt_overlapping_indices = pc_intersection(
        src_points, tgt_points, radius
    )
    
    # Calculate total overlapping points (union of both sets)
    total_overlapping = len(src_overlapping_indices) + len(tgt_overlapping_indices)
    total_points = len(src_points) + len(tgt_points)
    
    # Calculate overlap ratio
    overlap_ratio = total_overlapping / total_points if total_points > 0 else 0
    
    return overlap_ratio
