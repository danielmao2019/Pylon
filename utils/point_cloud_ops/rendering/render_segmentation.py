"""Segmentation rendering from point clouds using projection."""

import torch
from typing import Dict, Tuple, Union


def render_segmentation_from_pointcloud(
    pc_data: Dict[str, torch.Tensor],
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = "opengl",
    ignore_index: int = 255,
    key: str = "labels",
    return_mask: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render segmentation map from point cloud using camera projection.

    Projects 3D point cloud coordinates with segmentation labels onto 2D image
    plane using camera parameters. Creates a pixel-wise segmentation map where
    each pixel contains the label of the closest projected point.

    Args:
        pc_data: Point cloud dictionary containing 'pos' key with 3D coordinates
                 and segmentation labels under the specified key
        camera_intrinsics: 3x3 camera intrinsics matrix with focal lengths and principal point
        camera_extrinsics: 4x4 camera extrinsics matrix (camera-to-world transform)
        resolution: Target resolution as (width, height) tuple - intrinsics scaled automatically
        convention: Camera extrinsics convention ("opengl" supported, "standard" not implemented)
        ignore_index: Fill value for pixels with no point projections (default: 255)
        key: Key name for segmentation labels in pc_data (default: "labels")
        return_mask: If True, also return valid pixel mask (default: False)

    Returns:
        If return_mask is False:
            Segmentation map tensor of shape [H, W] with integer labels
        If return_mask is True:
            Tuple of (segmentation map tensor, valid mask tensor of shape [H, W] with boolean values)

    Raises:
        AssertionError: If point cloud is empty, labels are missing, or no points project within bounds
        NotImplementedError: If convention other than "opengl" is specified
    """
    # Step 1: Input validation
    assert isinstance(pc_data, dict), f"pc_data must be dict, got {type(pc_data)}"
    assert 'pos' in pc_data, f"pc_data must contain 'pos' key, got keys: {list(pc_data.keys())}"
    assert key in pc_data, f"pc_data must contain '{key}' key, got keys: {list(pc_data.keys())}"

    points = pc_data['pos']
    labels = pc_data[key]

    # Validate points tensor
    assert isinstance(points, torch.Tensor), f"points must be torch.Tensor, got {type(points)}"
    assert points.ndim == 2, f"points must be 2D tensor with shape (N, 3), got shape {points.shape}"
    assert points.shape[1] == 3, f"points must have 3 coordinates (XYZ), got shape {points.shape}"
    assert points.shape[0] > 0, f"Point cloud cannot be empty, got {points.shape[0]} points"

    # Validate segmentation labels tensor
    assert isinstance(labels, torch.Tensor), f"labels must be torch.Tensor, got {type(labels)}"
    assert labels.ndim == 1, f"labels must be 1D tensor with shape (N,), got shape {labels.shape}"
    assert labels.shape[0] == points.shape[0], f"labels length {labels.shape[0]} != points length {points.shape[0]}"

    # Validate camera matrices
    assert isinstance(camera_intrinsics, torch.Tensor), f"camera_intrinsics must be torch.Tensor, got {type(camera_intrinsics)}"
    assert camera_intrinsics.shape == (3, 3), f"camera_intrinsics must be 3x3 matrix, got shape {camera_intrinsics.shape}"
    assert isinstance(camera_extrinsics, torch.Tensor), f"camera_extrinsics must be torch.Tensor, got {type(camera_extrinsics)}"
    assert camera_extrinsics.shape == (4, 4), f"camera_extrinsics must be 4x4 matrix, got shape {camera_extrinsics.shape}"

    # Validate all tensors are on same device
    assert points.device == camera_intrinsics.device, f"points device {points.device} != camera_intrinsics device {camera_intrinsics.device}"
    assert points.device == camera_extrinsics.device, f"points device {points.device} != camera_extrinsics device {camera_extrinsics.device}"
    assert points.device == labels.device, f"points device {points.device} != labels device {labels.device}"

    # Validate resolution
    assert isinstance(resolution, (tuple, list)), f"resolution must be tuple or list, got {type(resolution)}"
    assert len(resolution) == 2, f"resolution must have 2 elements (width, height), got {len(resolution)}"
    assert all(isinstance(x, int) and x > 0 for x in resolution), f"resolution must be positive integers, got {resolution}"

    # Validate convention
    assert isinstance(convention, str), f"convention must be str, got {type(convention)}"
    assert convention in ["opengl", "standard"], f"convention must be 'opengl' or 'standard', got '{convention}'"

    # Validate ignore_index
    assert isinstance(ignore_index, int), f"ignore_index must be int, got {type(ignore_index)}"
    assert 0 <= ignore_index <= 255, f"ignore_index must be in range [0, 255], got {ignore_index}"

    # Validate key parameter
    assert isinstance(key, str), f"key must be str, got {type(key)}"

    # Validate return_mask
    assert isinstance(return_mask, bool), f"return_mask must be bool, got {type(return_mask)}"

    render_width, render_height = resolution

    # Step 2: Device and dtype conversions
    points = points.clone().to(dtype=torch.float64)
    camera_intrinsics = camera_intrinsics.clone().to(device=points.device, dtype=torch.float64)
    camera_extrinsics = camera_extrinsics.clone().to(device=points.device, dtype=torch.float64)

    # Step 3: Scale camera intrinsics in-place
    original_width = int(camera_intrinsics[0, 2] * 2)   # Estimate from principal point cx
    original_height = int(camera_intrinsics[1, 2] * 2)  # Estimate from principal point cy
    scale_x = render_width / original_width
    scale_y = render_height / original_height

    camera_intrinsics[0, 0] *= scale_x  # Scale focal length fx
    camera_intrinsics[1, 1] *= scale_y  # Scale focal length fy
    camera_intrinsics[0, 2] *= scale_x  # Scale principal point cx
    camera_intrinsics[1, 2] *= scale_y  # Scale principal point cy

    # Assert last row is [0, 0, 1] to ensure depth preservation during projection
    assert torch.allclose(
        camera_intrinsics[2, :],
        torch.tensor([0.0, 0.0, 1.0], device=camera_intrinsics.device, dtype=camera_intrinsics.dtype)
    ), f"Camera intrinsics last row must be [0, 0, 1], got {camera_intrinsics[2, :]}"

    # Step 4: Camera convention conversion
    if convention == "opengl":
        # Convert to world-to-camera by inverting camera-to-world extrinsics
        # Use materialize_tensor to avoid lazy wrapper issues
        from utils.ops.materialize_tensor import materialize_tensor
        materialized_extrinsics = materialize_tensor(camera_extrinsics)
        world_to_camera = torch.inverse(materialized_extrinsics)
    else:
        raise NotImplementedError("Standard convention not implemented yet")

    # Step 5: Transform points into camera local frame
    # Use addmm for memory efficiency: points @ R.T + t
    points = torch.addmm(world_to_camera[:3, 3], points, world_to_camera[:3, :3].T)

    # Step 6: Filter points based on z coordinate (OpenGL: negative Z in front)
    depth_mask = points[:, 2] < 0
    points = points[depth_mask]
    labels = labels[depth_mask]
    assert len(points) > 0, f"No points in front of camera for segmentation rendering, got {len(points)} valid points"

    # Step 7: Project points into 2D
    # Matrix multiplication preserves depth column due to [0, 0, 1] row assertion above
    points = (camera_intrinsics @ points.T).T

    # Perspective division and horizontal flip
    points[:, 0] /= points[:, 2]  # Perspective division for x
    points[:, 1] /= points[:, 2]  # Perspective division for y
    points[:, 0] = (render_width - 1) - points[:, 0]  # Apply horizontal flip correction

    # Step 8: Filter points within image bounds
    bounds_mask = (
        (points[:, 0] >= 0) & (points[:, 0] < render_width) &
        (points[:, 1] >= 0) & (points[:, 1] < render_height)
    )
    points = points[bounds_mask]
    labels = labels[bounds_mask]
    assert len(points) > 0, f"No points within image bounds for segmentation rendering, got {len(points)} valid points"

    # Step 9: Sort points by depth (closest first)
    sort_indices = torch.argsort(torch.abs(points[:, 2]))
    points = points[sort_indices]
    labels = labels[sort_indices]

    # Step 10: Allocate segmentation map
    seg_map = torch.full(
        (render_height, render_width),
        ignore_index,
        dtype=labels.dtype,
        device=labels.device
    )

    # Step 11: Fill segmentation map with sorted labels
    pixel_coords_x = points[:, 0].long()
    pixel_coords_y = points[:, 1].long()
    seg_map[pixel_coords_y, pixel_coords_x] = labels

    # Convert to int64 for final output
    seg_map = seg_map.to(dtype=torch.int64)

    if return_mask:
        # Step 12: Create valid mask
        valid_mask = torch.zeros(
            (render_height, render_width),
            dtype=torch.bool,
            device=points.device
        )
        valid_mask[pixel_coords_y, pixel_coords_x] = True
        
        return seg_map, valid_mask
    else:
        return seg_map
