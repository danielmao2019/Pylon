from typing import Optional
import numpy as np
import torch
from scipy.spatial import cKDTree
import open3d as o3d
from utils.point_cloud_ops.apply_transform import apply_transform


def get_correspondences(ref_points: torch.Tensor, src_points: torch.Tensor, transform: torch.Tensor, matching_radius: float) -> torch.Tensor:
    """Find correspondences between two point clouds within a matching radius.

    Args:
        ref_points (torch.Tensor): Reference point cloud [N, 3]
        src_points (torch.Tensor): Source point cloud [M, 3]
        transform (torch.Tensor): Transformation matrix from source to reference [4, 4]
        matching_radius (float): Maximum distance threshold for correspondence matching

    Returns:
        torch.Tensor: Correspondence indices [K, 2] where K is number of correspondences
    """
    # Convert to numpy for scipy operations
    ref_points = ref_points.cpu().numpy()
    src_points = src_points.cpu().numpy()
    transform = transform.cpu().numpy()

    # Transform source points to reference frame
    src_points = apply_transform(src_points, transform)

    # Build KD-tree for efficient search
    src_tree = cKDTree(src_points)

    # Find correspondences within radius
    indices_list = src_tree.query_ball_point(ref_points, matching_radius)

    # Create correspondence pairs
    corr_indices = np.array([
        (i, j)
        for i, indices in enumerate(indices_list)
        for j in indices
    ], dtype=np.int64)

    return torch.from_numpy(corr_indices)


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
