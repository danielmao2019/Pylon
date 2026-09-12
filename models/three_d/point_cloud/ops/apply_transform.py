from typing import Optional, Tuple, Union

import numpy as np
import torch

from utils.ops.chunked_matmul import chunked_matmul


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
        inplace: If True, the transformed coordinates are copied back into points' own buffer; if False, a new array/tensor is returned. Requires a transform carrying no leading batch axes, since those yield one copy of the points per entry and leave no single buffer to write back into.
        max_divide: Maximum number of times the torch matmul may halve its row batch on CUDA OOM (forwarded to chunked_matmul); ignored on the numpy path.
        num_divide: If not None, the fixed number of halvings for the torch matmul row batch (forwarded to chunked_matmul); ignored on the numpy path.

    Returns:
        Transformed points as the same type as points, numpy.ndarray or torch.Tensor of shape [N, 3] or [1, N, 3], gaining the transform's leading batch axes as [..., N, 3] when the transform carries any. When inplace, the result lives in points' own buffer: the same object as points when points is [N, 3], a [1, N, 3] view of it when points is batched.
    """

    def _validate_inputs() -> None:
        assert isinstance(points, (np.ndarray, torch.Tensor)), (
            "Expected points to be a numpy.ndarray or a torch.Tensor. "
            f"{type(points)=}"
        )
        assert (points.ndim == 2 and points.shape[1] == 3) or (
            points.ndim == 3 and points.shape[0] == 1 and points.shape[2] == 3
        ), (
            "Expected points to be [N, 3], or [1, N, 3] carrying one leading batch "
            f"axis of one. {points.shape=}"
        )
        assert isinstance(transform, (list, np.ndarray, torch.Tensor)), (
            "Expected transform to be a list, a numpy.ndarray or a torch.Tensor. "
            f"{type(transform)=}"
        )
        assert len(np.shape(transform)) >= 2 and tuple(np.shape(transform)[-2:]) == (
            4,
            4,
        ), (
            "Transform must be of shape [4, 4], optionally with leading batch axes. "
            f"{np.shape(transform)=}"
        )
        assert isinstance(inplace, bool), (
            "Expected inplace to be a bool. " f"{type(inplace)=}"
        )
        if inplace:
            # Leading axes survive normalization, so even a [1, 4, 4] transform genuinely yields its own copy of the points and cannot be written back into points.
            assert len(np.shape(transform)) == 2, (
                "inplace=True requires a transform with no leading axes: they yield one "
                "copy of the points per entry, leaving no single buffer to write back "
                f"into. {np.shape(transform)=}"
            )

    _validate_inputs()

    def _normalize_inputs(
        points: Union[np.ndarray, torch.Tensor],
        transform: Union[list, np.ndarray, torch.Tensor],
    ) -> Tuple[Union[np.ndarray, torch.Tensor], bool, Union[np.ndarray, torch.Tensor]]:
        # An unbatched [N, 3], a view of the caller's array when a batch axis was squeezed.
        points, was_batched = _normalize_points(points=points)
        transform = _normalize_transform(
            transform=transform,
            target_type=type(points),
            target_dtype=points.dtype,
            target_device=points.device if isinstance(points, torch.Tensor) else None,
        )
        assert transform.dtype == points.dtype, (
            "Expected the normalized points and transform to share one dtype. "
            f"{points.dtype=} {transform.dtype=}"
        )
        return points, was_batched, transform

    points, was_batched, transform = _normalize_inputs(
        points=points, transform=transform
    )

    if isinstance(points, np.ndarray):
        # Broadcasts over the transform's leading axes: [..., 4, 4] yields [..., N, 3], [4, 4] still [N, 3].
        transformed = np.matmul(
            np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)]),
            np.swapaxes(transform, -1, -2),
        )[..., :3]
        if inplace:
            # points is the caller's array, or a view of it when a batch axis was squeezed, so the write lands in the caller's own buffer.
            points[...] = transformed
            transformed = points
        if was_batched:
            transformed = np.expand_dims(transformed, axis=0)
        return transformed
    else:
        points_h = torch.cat(
            [
                points,
                torch.ones(
                    (points.shape[0], 1), dtype=points.dtype, device=points.device
                ),
            ],
            dim=1,
        )
        # Chunked over the point rows, broadcast over the transform's leading axes: [..., 4, 4] yields [..., N, 4], [4, 4] still [N, 4].
        transformed = chunked_matmul(
            large=points_h,
            small=transform.transpose(-1, -2),
            max_divide=max_divide,
            num_divide=num_divide,
        )[..., :3]
        if inplace:
            # points is the caller's tensor, or a view of it when a batch axis was squeezed, so the write lands in the caller's own buffer.
            points.copy_(transformed)
            transformed = points
        if was_batched:
            transformed = transformed.unsqueeze(0)
        return transformed


def _normalize_points(
    points: Union[np.ndarray, torch.Tensor],
) -> Tuple[Union[np.ndarray, torch.Tensor], bool]:
    """Normalize points to unbatched format (N, 3) while preserving type, reporting whether the input was batched.

    Args:
        points: Input points, numpy.ndarray or torch.Tensor of shape [N, 3] or [1, N, 3].

    Returns:
        Tuple of (normalized_points, was_batched) where:
        - normalized_points: Points with shape (N, 3), same type as input; a view of points when a batch axis was squeezed.
        - was_batched: True if input was batched [1, N, 3], False otherwise
    """
    if points.ndim == 2:
        normalized_points = points
        was_batched = False
        return normalized_points, was_batched
    else:
        # A view, so an inplace write through it lands in the caller's array.
        normalized_points = points.squeeze(0)
        was_batched = True
        return normalized_points, was_batched


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
