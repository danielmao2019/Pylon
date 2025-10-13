from typing import Optional, Union, Dict
import torch
from utils.point_cloud_ops.apply_transform import apply_transform
from utils.point_cloud_ops.knn.knn import knn
from utils.input_checks.check_point_cloud import check_point_cloud


def get_correspondences(
    source: Union[torch.Tensor, Dict[str, torch.Tensor]],
    target: Union[torch.Tensor, Dict[str, torch.Tensor]],
    transform: Optional[torch.Tensor],
    radius: float,
) -> torch.Tensor:
    """Find correspondences between two point clouds within a matching radius.

    Args:
        source (torch.Tensor or Dict[str, torch.Tensor]): Source point cloud [M, 3]
        target (torch.Tensor or Dict[str, torch.Tensor]): Target point cloud [N, 3]
        transform (torch.Tensor): Transformation matrix from source to target [4, 4] or None
        radius (float): Maximum distance threshold for correspondence matching

    Returns:
        torch.Tensor: Correspondence indices [K, 2] where K is number of correspondences
    """
    # Validate inputs
    if isinstance(source, dict):
        check_point_cloud(source)
        src_points = source['pos']
    else:
        src_points = source
    if isinstance(target, dict):
        check_point_cloud(target)
        tgt_points = target['pos']
    else:
        tgt_points = target

    assert (
        src_points.device == tgt_points.device
    ), f"{src_points.device=}, {tgt_points.device=}"
    assert transform is None or (
        isinstance(transform, torch.Tensor) and transform.shape == (4, 4)
    ), f"Invalid transform shape: {transform.shape if transform is not None else None}"
    assert (
        isinstance(radius, (int, float)) and radius > 0
    ), f"radius must be positive number, got {radius}"

    # Transform source points to reference frame if transform provided
    if transform is not None:
        # apply_transform supports torch tensors and will keep them as torch
        src_points = apply_transform(src_points, transform)

    # Use KNN to find all points within radius (no k limit, just radius)
    # Original algorithm: for each target point, find all source points within radius
    # This is equivalent to query=tgt_points, reference=src_points, radius=radius, k=None
    distances, indices = knn(
        query_points=tgt_points,
        reference_points=src_points,
        k=None,  # Return all neighbors within radius
        return_distances=True,
        radius=radius,
        method='scipy',
    )

    # Create correspondence pairs using vectorized operations
    # distances and indices are [N_tgt, k] where k = src_points.shape[0]
    # We need to convert to correspondence format [K, 2] where each row is [src_idx, tgt_idx]

    # Find valid correspondences (distance < inf and index >= 0)
    valid_mask = (distances < float('inf')) & (indices >= 0)

    # Get the coordinates of valid correspondences
    tgt_idx_coords, k_coords = torch.where(valid_mask)
    src_idx_coords = indices[tgt_idx_coords, k_coords]

    # Stack to create correspondence pairs [src_idx, tgt_idx]
    correspondences = torch.stack([src_idx_coords, tgt_idx_coords], dim=1)

    return correspondences
