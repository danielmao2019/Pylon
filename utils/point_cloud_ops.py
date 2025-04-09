from typing import List, Union, Tuple
import numpy as np
import torch
from scipy.spatial import cKDTree


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


def apply_transform(
    points: np.ndarray,
    transform: Union[List[List[Union[int, float]]], np.ndarray, torch.Tensor],
) -> torch.Tensor:
    """Apply 4x4 transformation matrix to points using homogeneous coordinates.

    Args:
        points (np.ndarray): Points to transform [N, 3]
        transform (Union[List[List[Union[int, float]]], np.ndarray, torch.Tensor]): 4x4 transformation matrix

    Returns:
        torch.Tensor: Transformed points [N, 3] with dtype=torch.float32
    """
    # Convert transform to torch.Tensor if it's not already
    if isinstance(transform, list):
        transform = torch.tensor(transform, dtype=torch.float32)
    elif isinstance(transform, np.ndarray):
        transform = torch.tensor(transform, dtype=torch.float32)
    else:
        raise ValueError(f"Transform must be a list or numpy array, got {type(transform)}")

    # Ensure transform is a 4x4 matrix
    assert transform.shape == (4, 4), f"Transform must be a 4x4 matrix, got {transform.shape}"

    # Convert points to torch tensor if it's not already
    if isinstance(points, np.ndarray):
        points = torch.tensor(points, dtype=torch.float32)
    
    # Add homogeneous coordinate
    points_h = torch.cat([points, torch.ones((points.shape[0], 1), dtype=torch.float32)], dim=1)

    # Apply transformation
    transformed = torch.matmul(points_h, transform.t())

    # Remove homogeneous coordinate
    return transformed[:, :3]


def compute_symmetric_difference_indices(
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
