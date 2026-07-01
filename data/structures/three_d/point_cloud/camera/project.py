import torch


def project_3d_to_2d(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """Project camera-frame 3D points to 2D pixel coordinates with a pinhole model.

    Assumes points are already in the camera local frame with +Z as the depth
    axis pointing away from the camera (OpenCV convention), then applies the
    pinhole projection u = fx * X / Z + cx, v = fy * Y / Z + cy.

    Args:
        points: Camera-frame 3D points as a float torch.Tensor of shape [N, 3],
            on the same device as intrinsics.
        intrinsics: Pinhole intrinsics matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            as a float torch.Tensor of shape [3, 3], on the same device as points.
        inplace: If False, return a new [N, 2] tensor. If True, overwrite the first
            two columns of points with the pixel coordinates and return points.

    Returns:
        If inplace is False, the [N, 2] pixel coordinates torch.Tensor. If inplace
        is True, the [N, 3] points tensor with its first two columns overwritten by
        the pixel coordinates.
    """
    # Input validation
    assert isinstance(points, torch.Tensor), (
        "Expected points to be a torch.Tensor. " f"{type(points)=}"
    )
    assert points.ndim == 2 and points.shape[1] == 3, (
        "Expected points to be a [N, 3] tensor. " f"{points.shape=}"
    )
    assert isinstance(intrinsics, torch.Tensor), (
        "Expected intrinsics to be a torch.Tensor. " f"{type(intrinsics)=}"
    )
    assert intrinsics.shape == (3, 3), (
        "Expected intrinsics to be a [3, 3] matrix. " f"{intrinsics.shape=}"
    )
    assert points.device == intrinsics.device, (
        "Expected points and intrinsics on the same device. "
        f"{points.device=} {intrinsics.device=}"
    )

    # Project 3D points to 2D using the pinhole camera model
    # Standard projection: u = fx * X / Z + cx, v = fy * Y / Z + cy
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    assert torch.all(points[:, 2] != 0), (
        "Cannot project points with a zero Z coordinate. "
        f"{int(torch.count_nonzero(points[:, 2] == 0).item())=}"
    )

    if inplace:
        # Pure in-place: modify points[:, 0] and points[:, 1] without temporaries
        points[:, 0].div_(points[:, 2])  # X = X / Z
        points[:, 0].mul_(fx).add_(cx)  # X = fx * X + cx

        points[:, 1].div_(points[:, 2])  # Y = Y / Z
        points[:, 1].mul_(fy).add_(cy)  # Y = fy * Y + cy

        return points
    else:
        # Pre-allocate and compute in-place for both columns
        result = torch.empty(
            points.shape[0], 2, dtype=points.dtype, device=points.device
        )
        result[:, 0] = points[:, 0]
        result[:, 0].div_(points[:, 2]).mul_(fx).add_(cx)  # X / Z * fx + cx
        result[:, 1] = points[:, 1]
        result[:, 1].div_(points[:, 2]).mul_(fy).add_(cy)  # Y / Z * fy + cy
        return result
