from typing import List, Union, Tuple
import numpy as np
import torch
from scipy.spatial import cKDTree


def apply_transform(
    points: Union[np.ndarray, torch.Tensor],
    transform: Union[List[List[Union[int, float]]], np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """Apply 4x4 transformation matrix to points using homogeneous coordinates.

    Args:
        points (Union[np.ndarray, torch.Tensor]): Points to transform [N, 3]
        transform (Union[List[List[Union[int, float]]], np.ndarray, torch.Tensor]): 4x4 transformation matrix

    Returns:
        Union[np.ndarray, torch.Tensor]: Transformed points [N, 3] with the same type as input points
    """
    # Determine if we're working with numpy or torch
    is_numpy = isinstance(points, np.ndarray)
    
    # Convert transform to the appropriate type
    if isinstance(transform, list):
        if is_numpy:
            transform = np.array(transform, dtype=np.float32)
        else:
            transform = torch.tensor(transform, dtype=torch.float32)
    elif isinstance(transform, np.ndarray):
        if not is_numpy:
            transform = torch.tensor(transform, dtype=torch.float32)
    elif isinstance(transform, torch.Tensor):
        if is_numpy:
            transform = transform.cpu().numpy()
    else:
        raise ValueError(f"Transform must be a list, numpy array, or torch tensor, got {type(transform)}")

    # Ensure transform is a 4x4 matrix
    assert transform.shape == (4, 4), f"Transform must be a 4x4 matrix, got {transform.shape}"
    
    # Add homogeneous coordinate
    if is_numpy:
        points_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
        
        # Apply transformation
        transformed = np.dot(points_h, transform.T)
        
        # Remove homogeneous coordinate
        return transformed[:, :3]
    else:
        points_h = torch.cat([points, torch.ones((points.shape[0], 1), dtype=torch.float32, device=points.device)], dim=1)
        
        # Apply transformation
        transformed = torch.matmul(points_h, transform.t())
        
        # Remove homogeneous coordinate
        return transformed[:, :3]
