"""Generate normal maps from depth maps using cross product approach."""

import torch
import torch.nn.functional as F


def depth_to_normals(
    depth_map: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    depth_ignore_value: float = float('inf'),
    normal_ignore_value: float = 0.0,
    return_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Generate normal map from depth map using cross product approach.

    Computes surface normals by converting depth to 3D points, then using
    cross product of vectors to neighboring pixels to compute normals.

    Output is in OpenCV camera coordinate system:
    - X-axis: Right (positive X points to the right in the image)
    - Y-axis: Down (positive Y points down in the image)
    - Z-axis: Forward (positive Z points into the scene/away from camera)

    Args:
        depth_map: Depth map tensor of shape [H, W] with depth values
        camera_intrinsics: 3x3 camera intrinsics matrix with focal lengths and principal point
        depth_ignore_value: Value representing invalid/missing depth in input (default: inf)
        normal_ignore_value: Value to use for invalid pixels in output normal map (default: 0.0)
        return_mask: If True, return tuple of (normal_map, mask). If False, return only normal_map

    Returns:
        If return_mask=False: Normal map tensor of shape [3, H, W] with normalized normal vectors in OpenCV
        camera coordinates. Invalid pixels (where depth == depth_ignore_value) have normal_ignore_value.
        If return_mask=True: Tuple of (normal_map, mask) where mask is boolean tensor of shape [H, W]
        indicating valid pixels.

    Example:
        >>> depth = torch.randn(480, 640)
        >>> K = torch.eye(3)
        >>> K[0, 0], K[1, 1] = 525.0, 525.0  # fx, fy
        >>> K[0, 2], K[1, 2] = 320.0, 240.0  # cx, cy
        >>> normals = depth_to_normals(depth, K)
        >>> print(normals.shape)  # torch.Size([3, 480, 640])
    """
    assert isinstance(
        depth_map, torch.Tensor
    ), f"depth_map must be torch.Tensor, got {type(depth_map)}"
    assert (
        depth_map.ndim == 2
    ), f"depth_map must be 2D tensor [H, W], got shape {depth_map.shape}"
    assert isinstance(
        camera_intrinsics, torch.Tensor
    ), f"camera_intrinsics must be torch.Tensor, got {type(camera_intrinsics)}"
    assert camera_intrinsics.shape == (
        3,
        3,
    ), f"camera_intrinsics must be 3x3 matrix, got shape {camera_intrinsics.shape}"
    assert isinstance(
        depth_ignore_value, (int, float)
    ), f"depth_ignore_value must be numeric, got {type(depth_ignore_value)}"
    assert isinstance(
        normal_ignore_value, (int, float)
    ), f"normal_ignore_value must be numeric, got {type(normal_ignore_value)}"

    H, W = depth_map.shape
    device = depth_map.device

    # Extract camera parameters
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    assert fx > 0, f"fx must be positive, got {fx}"
    assert fy > 0, f"fy must be positive, got {fy}"

    # Create pixel coordinate grids
    u, v = torch.meshgrid(
        torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy'
    )
    u = u.float()  # [H, W]
    v = v.float()  # [H, W]

    # Create mask for valid depth values
    if torch.isinf(torch.tensor(depth_ignore_value)):
        valid_mask = torch.isfinite(depth_map)
    else:
        valid_mask = depth_map != depth_ignore_value

    # Convert depth map to 3D points in camera coordinates
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    Z = depth_map
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Compute 3D points for neighboring pixels to get proper cross product

    # Right neighbor (u+1)
    u_right = torch.clamp(u + 1, max=W - 1)
    Z_right = torch.zeros_like(depth_map)
    Z_right[:, :-1] = depth_map[:, 1:]  # Shift depth values left
    Z_right[:, -1] = depth_map[:, -1]  # Replicate last column
    X_right = (u_right - cx) * Z_right / fx
    Y_right = (v - cy) * Z_right / fy

    # Down neighbor (v+1)
    v_down = torch.clamp(v + 1, max=H - 1)
    Z_down = torch.zeros_like(depth_map)
    Z_down[:-1, :] = depth_map[1:, :]  # Shift depth values up
    Z_down[-1, :] = depth_map[-1, :]  # Replicate last row
    X_down = (u - cx) * Z_down / fx
    Y_down = (v_down - cy) * Z_down / fy

    # Compute vectors from current pixel to neighbors
    vec_right = torch.stack([X_right - X, Y_right - Y, Z_right - Z], dim=0)  # [3, H, W]
    vec_down = torch.stack([X_down - X, Y_down - Y, Z_down - Z], dim=0)  # [3, H, W]

    # Compute normal via cross product: vec_right × vec_down
    normals = torch.cross(vec_right, vec_down, dim=0)

    # Normalize to unit vectors PER PIXEL (dim=0 normalizes across channels for each pixel)
    normals = F.normalize(normals, dim=0)

    # Create mask for pixels that should be invalid
    # Include pixels where depth differences couldn't be computed properly
    invalid_mask = ~valid_mask

    # Check for NaN values in computed normals (can happen at boundaries or discontinuities)
    nan_mask = torch.isnan(normals).any(dim=0)
    final_invalid_mask = invalid_mask | nan_mask

    # Set invalid pixels to normal_ignore_value
    normals[:, final_invalid_mask] = normal_ignore_value

    if return_mask:
        return normals, ~final_invalid_mask
    else:
        return normals
