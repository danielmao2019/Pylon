import numpy as np
import torch
from scipy.spatial import cKDTree
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
