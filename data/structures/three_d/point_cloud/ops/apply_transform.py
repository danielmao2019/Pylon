from typing import List, Tuple, Union, Optional

import numpy as np
import torch


def _normalize_points(
    points: Union[np.ndarray, torch.Tensor],
) -> Tuple[Union[np.ndarray, torch.Tensor], bool]:
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
        assert (
            points.shape[1] == 3
        ), f"Points must have 3 coordinates, got shape {points.shape}"
        return points, False
    elif points.ndim == 3:
        # Points are batched [B, N, 3]
        assert points.shape[0] == 1, f"Batch size must be 1, got shape {points.shape}"
        assert (
            points.shape[2] == 3
        ), f"Points must have 3 coordinates, got shape {points.shape}"

        # Squeeze batch dimension
        return points.squeeze(0), True
    else:
        raise ValueError(
            f"Points must have 2 or 3 dimensions, got shape {points.shape}"
        )


def _normalize_transform(
    transform: Union[list, np.ndarray, torch.Tensor],
    target_type: type,
    target_dtype: Union[torch.dtype, np.dtype],
    target_device: Optional[Union[str, torch.device]],
) -> Union[np.ndarray, torch.Tensor]:
    if target_type == np.ndarray:
        transform = _normalize_transform_numpy(
            transform=transform, target_dtype=target_dtype
        )
    elif target_type == torch.Tensor:
        transform = _normalize_transform_torch(
            transform=transform, target_dtype=target_dtype, target_device=target_device
        )
    else:
        raise ValueError(f"Unsupported target type: {target_type}")
    transform = transform.squeeze()
    assert transform.shape == (
        4,
        4,
    ), f"Transform must be of shape [4, 4], got {transform.shape}"
    return transform


def _normalize_transform_numpy(
    transform: Union[list, np.ndarray, torch.Tensor], target_dtype: np.dtype
) -> np.ndarray:
    if isinstance(transform, list):
        transform = np.array(transform, dtype=target_dtype)
    if isinstance(transform, torch.Tensor):
        transform = transform.cpu().numpy()
    return transform.astype(target_dtype)


def _normalize_transform_torch(
    transform: Union[list, np.ndarray, torch.Tensor],
    target_dtype: torch.dtype,
    target_device: torch.device,
) -> torch.Tensor:
    if isinstance(transform, list):
        transform = torch.tensor(transform, dtype=target_dtype, device=target_device)
    if isinstance(transform, np.ndarray):
        transform = torch.from_numpy(transform)
    return transform.to(dtype=target_dtype, device=target_device)


def apply_transform(
    points: Union[np.ndarray, torch.Tensor],
    transform: Union[list, np.ndarray, torch.Tensor],
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
    target_dtype = points_normalized.dtype
    target_device = (
        points_normalized.device
        if isinstance(points_normalized, torch.Tensor)
        else None
    )
    transform_normalized = _normalize_transform(
        transform=transform,
        target_type=target_type,
        target_dtype=target_dtype,
        target_device=target_device,
    )

    assert (
        points_normalized.dtype == transform_normalized.dtype
    ), f"Dtype mismatch: points={points_normalized.dtype}, transform={transform_normalized.dtype}"

    # Apply transformation using homogeneous coordinates
    if isinstance(points_normalized, np.ndarray):
        # Add homogeneous coordinate
        ones_column = np.ones(
            (points_normalized.shape[0], 1), dtype=points_normalized.dtype
        )
        points_h = np.hstack([points_normalized, ones_column])

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
        ones_column = torch.ones(
            (points_normalized.shape[0], 1),
            dtype=points_normalized.dtype,
            device=points_normalized.device,
        )
        points_h = torch.cat([points_normalized, ones_column], dim=1)

        # Apply transformation
        transformed = torch.matmul(points_h, transform_normalized.t())

        # Remove homogeneous coordinate
        result = transformed[:, :3]

        # Restore batch dimension if needed
        if points_was_batched:
            result = result.unsqueeze(0)

        return result
