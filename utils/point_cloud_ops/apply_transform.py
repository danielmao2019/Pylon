from typing import List, Union, Tuple
import numpy as np
import torch


def _normalize_points(points: Union[np.ndarray, torch.Tensor]) -> Tuple[Union[np.ndarray, torch.Tensor], bool]:
    """Normalize points to unbatched format (N, 3) while preserving type.
    
    Args:
        points: Input points, either [N, 3] or [1, N, 3]
        
    Returns:
        Tuple of (normalized_points, was_batched) where:
        - normalized_points: Points with shape (N, 3), same type as input
        - was_batched: True if input was batched [1, N, 3], False otherwise
    """
    if points.ndim == 2:
        # Points are unbatched [N, 3]
        assert points.shape[1] == 3, f"Points must have 3 coordinates, got shape {points.shape}"
        return points, False
    elif points.ndim == 3:
        # Points are batched [B, N, 3]
        assert points.shape[0] == 1, f"Batch size must be 1, got shape {points.shape}"
        assert points.shape[2] == 3, f"Points must have 3 coordinates, got shape {points.shape}"
        
        # Squeeze batch dimension
        if isinstance(points, np.ndarray):
            return points.squeeze(0), True
        else:  # torch.Tensor
            return points.squeeze(0), True
    else:
        raise ValueError(f"Points must have 2 or 3 dimensions, got shape {points.shape}")


def _normalize_transform(
    transform: Union[List[List[Union[int, float]]], np.ndarray, torch.Tensor],
    target_type: type,
    target_device: Union[str, torch.device, None] = None,
    target_dtype: Union[torch.dtype, np.dtype, None] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Normalize transform to unbatched (4, 4) format with target type, device, and dtype.
    
    Args:
        transform: Input transform, can be list, numpy array, or torch tensor
        target_type: Target type (np.ndarray or torch.Tensor)
        target_device: Target device for torch tensors (ignored for numpy arrays)
        target_dtype: Target data type (torch.dtype for tensors, np.dtype for arrays)
        
    Returns:
        Normalized transform with shape (4, 4) and target type/device/dtype
    """
    # Convert to tensor first
    if isinstance(transform, list):
        transform = np.array(transform, dtype=target_dtype)
    
    # Handle batched transforms
    if transform.ndim == 3:
        assert transform.shape == (1, 4, 4), f"Batched transform must have shape (1, 4, 4), got {transform.shape}"
        if isinstance(transform, np.ndarray):
            transform = transform.squeeze(0)
        else:  # torch.Tensor
            transform = transform.squeeze(0)
    elif transform.ndim == 2:
        assert transform.shape == (4, 4), f"Transform must have shape (4, 4), got {transform.shape}"
    else:
        raise ValueError(f"Transform must have 2 or 3 dimensions, got shape {transform.shape}")
    
    # Convert to target type with specified dtype
    if target_type == np.ndarray:
        # Convert to numpy array
        if isinstance(transform, torch.Tensor):
            numpy_transform = transform.cpu().numpy()
        else:
            numpy_transform = transform
        
        # Apply target dtype if specified, otherwise use float32
        if target_dtype is not None:
            return numpy_transform.astype(target_dtype)
        else:
            return numpy_transform.astype(np.float32)
    else:  # target_type == torch.Tensor
        # Convert to torch tensor
        if isinstance(transform, np.ndarray):
            # Apply target dtype if specified, otherwise use float32
            if target_dtype is not None:
                tensor = torch.tensor(transform, dtype=target_dtype)
            else:
                tensor = torch.tensor(transform, dtype=torch.float32)
        else:
            # Apply target dtype if specified, otherwise use float32
            if target_dtype is not None:
                tensor = transform.to(target_dtype)
            else:
                tensor = transform.to(torch.float32)
        
        # Apply target device if specified
        if target_device is not None:
            tensor = tensor.to(target_device)
        return tensor


def apply_transform(
    points: Union[np.ndarray, torch.Tensor],
    transform: Union[List[List[Union[int, float]]], np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """Apply 4x4 transformation matrix to points using homogeneous coordinates.

    Args:
        points: Points to transform [N, 3] or batched [1, N, 3]
        transform: 4x4 transformation matrix [4, 4] or batched [1, 4, 4]

    Returns:
        Transformed points [N, 3] or [1, N, 3] with the same type as input points
    """
    # Normalize points to unbatched format
    points_normalized, points_was_batched = _normalize_points(points)
    
    # Normalize transform to target type, device, and dtype matching points
    target_type = type(points_normalized)
    target_device = points_normalized.device if isinstance(points_normalized, torch.Tensor) else None
    target_dtype = points_normalized.dtype
    transform_normalized = _normalize_transform(transform, target_type, target_device, target_dtype)
    
    assert points_normalized.dtype == transform_normalized.dtype, f"{points_normalized.dtype=}, {transform_normalized.dtype=}"
    
    # Apply transformation using homogeneous coordinates
    if isinstance(points_normalized, np.ndarray):
        # Add homogeneous coordinate
        points_h = np.hstack([points_normalized, np.ones((points_normalized.shape[0], 1), dtype=np.float32)])
        
        # Apply transformation
        transformed = np.dot(points_h, transform_normalized.T)
        
        # Remove homogeneous coordinate
        result = transformed[:, :3]
        
        # Restore batch dimension if needed
        if points_was_batched:
            result = np.expand_dims(result, axis=0)
        
        return result
    else:  # torch.Tensor
        # Add homogeneous coordinate
        points_h = torch.cat([
            points_normalized, 
            torch.ones((points_normalized.shape[0], 1), dtype=torch.float32, device=points_normalized.device)
        ], dim=1)
        
        # Apply transformation
        transformed = torch.matmul(points_h, transform_normalized.t())
        
        # Remove homogeneous coordinate
        result = transformed[:, :3]
        
        # Restore batch dimension if needed
        if points_was_batched:
            result = result.unsqueeze(0)
        
        return result
