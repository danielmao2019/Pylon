from typing import Optional

import torch

from data.structures.three_d.point_cloud.ops.apply_transform import apply_transform


def world_to_camera_transform(
    points: torch.Tensor,
    extrinsics: torch.Tensor,
    inplace: bool = False,
    max_divide: int = 0,
    num_divide: Optional[int] = None,
) -> torch.Tensor:
    """Map world-frame points into the camera local frame.

    High-level API that builds the world-to-camera 4x4 matrix by inverting the
    camera-to-world extrinsic and applies it to the points via apply_transform.

    Args:
        points: Float torch.Tensor of shape [N, 3] in world coordinates, on the
            same device as extrinsics.
        extrinsics: Float torch.Tensor of shape [4, 4] representing the
            camera-to-world (pose) transform in the OpenCV convention, on the same
            device as points.
        inplace: If True, the camera-frame coordinates are written back into
            points and points is returned; if False, a new tensor is returned.
        max_divide: Maximum number of times the matmul may halve its row batch on
            CUDA OOM (forwarded to apply_transform).
        num_divide: If not None, the fixed number of halvings for the matmul row
            batch (forwarded to apply_transform).

    Returns:
        Float torch.Tensor of shape [N, 3] in the camera local frame (OpenCV:
        +Z forward). The same tensor as points when inplace.
    """

    def _validate_inputs() -> None:
        assert isinstance(points, torch.Tensor), (
            "Expected points to be a torch.Tensor. " f"{type(points)=}"
        )
        assert points.ndim == 2 and points.shape[1] == 3, (
            "Expected points to be a [N, 3] tensor. " f"{points.shape=}"
        )
        assert isinstance(extrinsics, torch.Tensor), (
            "Expected extrinsics to be a torch.Tensor. " f"{type(extrinsics)=}"
        )
        assert extrinsics.shape == (4, 4), (
            "Expected extrinsics to be a [4, 4] matrix. " f"{extrinsics.shape=}"
        )
        assert points.device == extrinsics.device, (
            "Expected points and extrinsics on the same device. "
            f"{points.device=} {extrinsics.device=}"
        )

    _validate_inputs()

    world_to_camera = torch.inverse(extrinsics)

    return apply_transform(
        points=points,
        transform=world_to_camera,
        inplace=inplace,
        max_divide=max_divide,
        num_divide=num_divide,
    )
