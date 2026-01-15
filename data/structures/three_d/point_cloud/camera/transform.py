import math

import torch

from data.structures.three_d.camera.validation import validate_camera_extrinsics
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from utils.ops.materialize_tensor import materialize_tensor


def _world_to_camera_transform(
    points: torch.Tensor, extrinsics: torch.Tensor, inplace: bool = False
) -> torch.Tensor:
    """Transform 3D points from world coordinates into the camera local frame.

    Assumes `extrinsics` is a camera-to-world (pose) 4x4 matrix in the OpenCV
    convention. Validates inputs, applies the transform, and returns the points
    in the camera frame.

    Args:
        points: Tensor of shape [N, 3].
        extrinsics: Tensor of shape [4, 4] representing camera-to-world transform.

    Returns:
        points_cam: [N, 3] points in camera local frame (OpenCV: +Z forward).
    """
    # Input validation
    # Validate points as XYZ coordinates
    assert isinstance(points, torch.Tensor), f"{type(points)=}"
    assert (
        points.shape[0] > 0
    ), f"Expected positive number of points, got {points.shape[0]}"
    PointCloud.validate_xyz_tensor(xyz=points)
    validate_camera_extrinsics(extrinsics)
    # Device compatibility: points and extrinsics must be on same device
    assert (
        points.device == extrinsics.device
    ), f"points device {points.device} != extrinsics device {extrinsics.device}"

    # Convert to world-to-camera by inverting camera-to-world extrinsics
    world_to_camera = torch.inverse(materialize_tensor(extrinsics))

    # Transform points into camera local frame (OpenCV convention) using addmm
    out = torch.addmm(world_to_camera[:3, 3], points, world_to_camera[:3, :3].T)
    if inplace:
        points.copy_(out)
        return points
    return out


def _world_to_camera_transform_batched(
    points: torch.Tensor, extrinsics: torch.Tensor, batch_size: int
) -> None:
    """Process points in fixed-size batches, writing results into `points`.

    Always calls `_world_to_camera_transform` with `inplace=True` on slices.

    Args:
        points: [N,3] tensor to transform, updated in-place.
        extrinsics: [4,4] camera-to-world transform.
        batch_size: positive int batch size.
    """
    N = points.shape[0]
    for i in range(0, N, batch_size):
        j = min(N, i + batch_size)
        _world_to_camera_transform(points[i:j], extrinsics, inplace=True)


def world_to_camera_transform(
    points: torch.Tensor,
    extrinsics: torch.Tensor,
    inplace: bool = False,
    max_divide: int = 0,
    num_divide: int | None = None,
) -> torch.Tensor:
    """Divide-and-conquer wrapper for world-to-camera transform to avoid CUDA OOM.

    Attempts to run `_world_to_camera_transform` on the full tensor. If a CUDA
    OOM occurs and `max_divide > 0`, iteratively halves the batch and processes
    chunks, writing results back into `points` (or a clone if not `inplace`).

    Args:
        points: Float tensor of shape [N, 3].
        extrinsics: Float tensor of shape [4, 4]. Same device as `points`.
        inplace: If True, writes results into `points` tensor.
        max_divide: Max number of times to split the points in half on OOM.

    Returns:
        Tensor of shape [N, 3] with camera-frame coordinates.
    """
    # Handle in-place semantics at the wrapper level
    if not inplace:
        points = points.clone()

    # If a fixed division is provided, run a single batched pass with that division
    N = points.shape[0]
    if num_divide is not None:
        batch_size = max(1, math.ceil(N / (2**num_divide)))
        _world_to_camera_transform_batched(points, extrinsics, batch_size)
        return points

    # Exponential halving strategy controlled by max_divide.
    n = 0
    while n <= max_divide:
        batch_size = max(1, math.ceil(N / (2**n)))
        try:
            _world_to_camera_transform_batched(points, extrinsics, batch_size)
            return points
        except torch.cuda.OutOfMemoryError:
            n += 1
            torch.cuda.empty_cache()
            continue
        except Exception:
            raise

    raise torch.cuda.OutOfMemoryError(
        "CUDA OOM after max_divide reductions in world_to_camera_transform"
    )
