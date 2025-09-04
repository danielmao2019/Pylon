import torch
from utils.input_checks.check_point_cloud import check_pc_xyz


def project_3d_to_2d(
    points: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """Project 3D points to 2D pixel coordinates using camera intrinsics.

    CRITICAL ASSUMPTIONS:
    1. Points are in camera local coordinate frame (not world coordinates)
    2. +Z is the depth direction pointing away from the camera (OpenCV convention)

    Args:
        points: 3D points in camera coordinate system, shape (N, 3)
        camera_intrinsics: Camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], shape (3, 3)
        inplace: If False, return (N, 2) tensor. If True, modify first two columns of points tensor

    Returns:
        If inplace=False: 2D pixel coordinates, shape (N, 2)
        If inplace=True: Modified points tensor with first two columns as 2D coordinates, shape (N, 3)
    """
    check_pc_xyz(points)
    assert isinstance(camera_intrinsics, torch.Tensor), f"camera_intrinsics must be a torch.Tensor, got {type(camera_intrinsics)}"
    assert camera_intrinsics.shape == (3, 3), f"camera_intrinsics must be a 3x3 matrix, got shape {camera_intrinsics.shape}"
    assert (
        camera_intrinsics[0, 1] == 0
        and camera_intrinsics[1, 0] == 0
        and camera_intrinsics[2, 0] == 0
        and camera_intrinsics[2, 1] == 0
        and camera_intrinsics[2, 2] == 1
    )
    assert camera_intrinsics.is_floating_point(), f"camera_intrinsics must be floating point, got {camera_intrinsics.dtype}"
    assert points.device == camera_intrinsics.device

    # Project 3D points to 2D using pinhole camera model
    # Standard projection: u = fx * X/Z + cx, v = fy * Y/Z + cy
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # Avoid division by zero
    assert torch.all(points[:, 2] != 0), "Cannot project points with zero Z coordinate"

    # Compute 2D pixel coordinates
    if inplace:
        # Pure in-place: modify points[:, 0] and points[:, 1] without temporary tensors
        points[:, 0].div_(points[:, 2])  # X = X / Z
        points[:, 0].mul_(fx).add_(cx)   # X = fx * X + cx

        points[:, 1].div_(points[:, 2])  # Y = Y / Z
        points[:, 1].mul_(fy).add_(cy)   # Y = fy * Y + cy

        return points
    else:
        # Pre-allocate and compute in-place for both columns
        result = torch.empty(points.shape[0], 2, dtype=points.dtype, device=points.device)
        result[:, 0] = points[:, 0]
        result[:, 0].div_(points[:, 2]).mul_(fx).add_(cx)  # X/Z * fx + cx
        result[:, 1] = points[:, 1]
        result[:, 1].div_(points[:, 2]).mul_(fy).add_(cy)  # Y/Z * fy + cy
        return result
