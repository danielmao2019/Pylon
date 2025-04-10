from typing import Tuple, List
import torch


def point_cloud_intersection(
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
    # Count source points that are close to any target point
    src_overlapping_indices = []
    for i, src_point in enumerate(src_points):
        # Find points in the target point cloud that are close to this source point
        distances = torch.norm(tgt_points - src_point, dim=1)
        close_points = torch.where(distances < radius)[0]
        if len(close_points) > 0:
            src_overlapping_indices.append(i)
    
    # Count target points that are close to any source point
    tgt_overlapping_indices = []
    for i, tgt_point in enumerate(tgt_points):
        # Find points in the source point cloud that are close to this target point
        distances = torch.norm(src_points - tgt_point, dim=1)
        close_points = torch.where(distances < radius)[0]
        if len(close_points) > 0:
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
    # Get overlapping indices
    src_overlapping_indices, tgt_overlapping_indices = point_cloud_intersection(
        src_points, tgt_points, radius
    )
    
    # Calculate total overlapping points (union of both sets)
    total_overlapping = len(src_overlapping_indices) + len(tgt_overlapping_indices)
    total_points = len(src_points) + len(tgt_points)
    
    # Calculate overlap ratio
    overlap_ratio = total_overlapping / total_points if total_points > 0 else 0
    
    return overlap_ratio
