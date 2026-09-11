from typing import Optional, Tuple, Union

import numpy as np
import torch

from utils.ops.chunked_matmul import chunked_matmul


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
    """Normalize a transform to the target type, dtype, and device, leaving its leading axes as they came in.

    Args:
        transform: 4x4 homogeneous transformation matrix applied to column-vector points (a point maps as transform @ [x, y, z, 1]), as a nested list, numpy.ndarray, or torch.Tensor of shape [4, 4], or a stack [..., 4, 4] carrying leading batch axes, in any dtype and, for a torch.Tensor, on any device.
        target_type: Type the transform is normalized to, either numpy.ndarray or torch.Tensor; any other type raises ValueError.
        target_dtype: Dtype the transform is cast to: a numpy dtype when target_type is numpy.ndarray, a torch.dtype when target_type is torch.Tensor.
        target_device: Device the transform is placed on when target_type is torch.Tensor; None when target_type is numpy.ndarray, where devices do not apply.

    Returns:
        The transform as target_type of shape [..., 4, 4] with dtype target_dtype, on target_device when target_type is torch.Tensor, its leading batch axes exactly as they came in: [4, 4] stays [4, 4] and [1, 4, 4] keeps its leading axis.
    """
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
    assert transform.ndim >= 2 and tuple(transform.shape[-2:]) == (
        4,
        4,
    ), f"Transform must be of shape [4, 4], optionally with leading batch axes, got {transform.shape}"
    return transform


def _normalize_transform_numpy(
    transform: Union[list, np.ndarray, torch.Tensor], target_dtype: np.dtype
) -> np.ndarray:
    """Convert a list or tensor transform into a numpy array of the target dtype.

    Args:
        transform: 4x4 homogeneous transformation matrix as a nested list, numpy.ndarray, or torch.Tensor of shape [4, 4], or a stack [..., 4, 4] carrying leading batch axes, in any dtype and, for a torch.Tensor, on any device (it is moved to the cpu before conversion).
        target_dtype: Numpy dtype the returned array is cast to.

    Returns:
        The transform as a numpy.ndarray of shape [..., 4, 4] and dtype target_dtype, its leading batch axes unchanged.
    """
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
    """Convert a list or ndarray transform into a torch tensor on the target dtype and device.

    Args:
        transform: 4x4 homogeneous transformation matrix as a nested list, numpy.ndarray, or torch.Tensor of shape [4, 4], or a stack [..., 4, 4] carrying leading batch axes, in any dtype and, for a torch.Tensor, on any device.
        target_dtype: Torch dtype the returned tensor is cast to.
        target_device: Torch device the returned tensor is placed on.

    Returns:
        The transform as a torch.Tensor of shape [..., 4, 4] with dtype target_dtype on target_device, its leading batch axes unchanged.
    """
    if isinstance(transform, list):
        transform = torch.tensor(transform, dtype=target_dtype, device=target_device)
    if isinstance(transform, np.ndarray):
        transform = torch.from_numpy(transform)
    return transform.to(dtype=target_dtype, device=target_device)


def apply_transform(
    points: Union[np.ndarray, torch.Tensor],
    transform: Union[list, np.ndarray, torch.Tensor],
    inplace: bool = False,
    max_divide: int = 0,
    num_divide: Optional[int] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Apply a 4x4 transformation matrix to points using homogeneous coordinates.

    Args:
        points: Points to transform, numpy.ndarray or torch.Tensor of shape [N, 3] or batched [1, N, 3], any float dtype. The type, dtype, and (for tensors) device of the output match this input.
        transform: 4x4 transformation matrix as a list, numpy.ndarray, or torch.Tensor of shape [4, 4], or a stack [..., 4, 4] carrying leading batch axes (a stack of one, [1, 4, 4], keeps its axis). Normalized to the type, dtype, and device of points, its leading axes left as they came in. A stack broadcasts over its leading axes, so every matrix it carries transforms the same [N, 3] points.
        inplace: If True, the transformed coordinates are copied back into points and points is returned; if False, a new array/tensor is returned. Requires a transform carrying no leading batch axes, since those yield one copy of the points per entry and leave no single buffer to write back into.
        max_divide: Maximum number of times the torch matmul may halve its row batch on CUDA OOM (forwarded to chunked_matmul); ignored on the numpy path.
        num_divide: If not None, the fixed number of halvings for the torch matmul row batch (forwarded to chunked_matmul); ignored on the numpy path.

    Returns:
        Transformed points as the same type as points, numpy.ndarray or torch.Tensor of shape [N, 3] or [1, N, 3], gaining the transform's leading batch axes as [..., N, 3] when the transform carries any. When inplace, this is the same object as points, which a transform carrying leading batch axes therefore cannot be.
    """

    def _validate_inputs() -> None:
        if inplace:
            # Leading axes survive normalization, so even a [1, 4, 4] transform genuinely yields its own copy of the points and cannot be written back into points.
            assert (
                len(np.shape(transform)) == 2
            ), f"inplace=True requires a transform with no leading axes: they yield one copy of the points per entry, leaving no single buffer to write back into, got {np.shape(transform)=}"

    _validate_inputs()

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

        # Apply transformation and remove homogeneous coordinate
        result = np.matmul(points_h, np.swapaxes(transform_normalized, -1, -2))[..., :3]

        # Restore batch dimension if needed
        if points_was_batched:
            result = np.expand_dims(result, axis=0)

        if inplace:
            points[...] = result
            return points
        return result
    else:  # torch.Tensor
        # Add homogeneous coordinate
        ones_column = torch.ones(
            (points_normalized.shape[0], 1),
            dtype=points_normalized.dtype,
            device=points_normalized.device,
        )
        points_h = torch.cat([points_normalized, ones_column], dim=1)

        # Apply transformation and remove homogeneous coordinate
        result = chunked_matmul(
            large=points_h,
            small=transform_normalized.transpose(-1, -2),
            max_divide=max_divide,
            num_divide=num_divide,
        )[..., :3]

        # Restore batch dimension if needed
        if points_was_batched:
            result = result.unsqueeze(0)

        if inplace:
            points.copy_(result)
            return points
        return result
