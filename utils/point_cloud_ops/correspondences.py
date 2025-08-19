from typing import Optional
import numpy as np
import torch
from scipy.spatial import cKDTree
import open3d as o3d
from utils.point_cloud_ops.apply_transform import apply_transform


def get_correspondences(src_points: torch.Tensor, tgt_points: torch.Tensor, transform: Optional[torch.Tensor], radius: float) -> torch.Tensor:
    """Find correspondences between two point clouds within a matching radius.

    Args:
        src_points (torch.Tensor): Source point cloud [M, 3]
        tgt_points (torch.Tensor): Target point cloud [N, 3]
        transform (torch.Tensor): Transformation matrix from source to target [4, 4] or None
        radius (float): Maximum distance threshold for correspondence matching

    Returns:
        torch.Tensor: Correspondence indices [K, 2] where K is number of correspondences
    """
    assert src_points.device == tgt_points.device, f"{src_points.device=}, {tgt_points.device=}"
    device = src_points.device

    # Convert to numpy for scipy operations
    tgt_points_np = tgt_points.cpu().numpy()
    src_points_np = src_points.cpu().numpy()
    if transform is not None:
        transform_np = transform.cpu().numpy()
    else:
        transform_np = None

    # Transform source points to reference frame
    if transform_np is not None:
        src_points_transformed = apply_transform(src_points_np, transform_np)
    else:
        src_points_transformed = src_points_np

    # Build KD-tree for efficient search
    src_tree = cKDTree(src_points_transformed)
    
    # Find correspondences within radius using chunked approach to avoid OOM
    # Chunked processing to avoid massive memory allocation
    chunk_size = 1000000  # 1M points per chunk
    total_tgt_points = len(tgt_points_np)
    
    # Process in chunks
    indices_list = []
    total_correspondences = 0
    
    num_chunks = (total_tgt_points + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_tgt_points)
        chunk_tgt_points = tgt_points_np[start_idx:end_idx]
        
        # Process this chunk
        chunk_indices = src_tree.query_ball_point(chunk_tgt_points, radius)
        
        # Count correspondences in this chunk
        chunk_corr_count = sum(len(indices) for indices in chunk_indices)
        total_correspondences += chunk_corr_count
        
        # Extend the global indices list
        indices_list.extend(chunk_indices)
    
    # Create correspondence pairs from chunked results
    corr_list = []
    processed_tgt_points = 0
    
    # Process chunks to create correspondence pairs with correct global indices
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_tgt_points)
        chunk_size_actual = end_idx - start_idx
        
        # Extract this chunk's indices
        chunk_indices = indices_list[processed_tgt_points:processed_tgt_points + chunk_size_actual]
        
        # Create correspondence pairs with global target indices
        chunk_corr_list = [
            (i, j + start_idx)  # j + start_idx gives global target index
            for j, indices in enumerate(chunk_indices)
            for i in indices
        ]
        
        corr_list.extend(chunk_corr_list)
        processed_tgt_points += chunk_size_actual
    
    # Handle empty case - ensure we get a 2D array with shape (0, 2)
    if len(corr_list) == 0:
        corr_indices = np.empty((0, 2), dtype=np.int64)
    else:
        corr_indices = np.array(corr_list, dtype=np.int64)

    result = torch.tensor(corr_indices, dtype=torch.int64, device=device)
    return result


def get_correspondences_v2(
    src_pcd: o3d.geometry.PointCloud,
    tgt_pcd: o3d.geometry.PointCloud,
    trans: np.ndarray,
    search_voxel_size: float,
    K: Optional[int] = None
) -> torch.Tensor:
    """Find correspondences between two point clouds within a matching radius.

    Args:
        src_pcd (o3d.geometry.PointCloud): Source point cloud
        tgt_pcd (o3d.geometry.PointCloud): Target point cloud
        trans (np.ndarray): Transformation matrix from source to target
        search_voxel_size (float): Search voxel size
    """
    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences
