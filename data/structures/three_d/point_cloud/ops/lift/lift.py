"""
Potential improvements:
1. Different fusion strategies: weighted voting, confidence-based fusion, bayesian fusion.
2. Multi-scale occlusion handling: consider point visibility at multiple scales.
3. Optimal view selection.
4. Batched processing.
"""

from typing import List, Tuple

import torch

from data.structures.three_d.camera.camera import Camera
from utils.input_checks.tensor_types import check_semantic_segmentation_pred
from utils.ops.materialize_tensor import materialize_tensor


def multi_view_fusion(
    points: torch.Tensor,  # [N, 3] 3D point coordinates
    maps: List[torch.Tensor],  # List of [N, C, H, W] segmentation probability maps
    cameras: List[Camera],  # List of Camera objects
    ignore_value: int = 255,  # Value to assign when no cameras vote
) -> torch.Tensor:
    """
    Fuse multi-view 2D segmentation maps to 3D point cloud using majority voting.

    Args:
        points: [N, 3] 3D point coordinates
        maps: List of [N, C, H, W] segmentation probability maps (N batch, C classes, H height, W width)
        cameras: List of Camera objects (camera-to-world transforms)
        ignore_value: Value to assign when no cameras vote for those points

    Returns:
        final_labels: [N] predicted class indices for each point (or ignore_value for unvoted points)
    """
    # Input validation
    assert isinstance(maps, list)
    [check_semantic_segmentation_pred(m) for m in maps]
    assert isinstance(cameras, list), f"{type(cameras)=}"
    assert all(
        isinstance(camera, Camera) for camera in cameras
    ), "All cameras must be Camera instances"
    assert len(maps) == len(cameras), (
        f"Length of maps and cameras must be the same. "
        f"Got {len(maps)} and {len(cameras)}"
    )
    assert all(
        [
            m.shape[-2] == int(camera.intrinsics[1, 2] * 2)
            and m.shape[-1] == int(camera.intrinsics[0, 2] * 2)
            for m, camera in zip(maps, cameras, strict=True)
        ]
    ), "Incompatible map and camera intrinsic dimensions."

    N = points.shape[0]  # N: number of points
    C = maps[0].shape[1]  # C: number of classes (maps are [N, C, H, W])

    # Initialize vote accumulation matrix
    vote_matrix = torch.zeros(N, C, device=points.device)  # [N, C] vote counts

    # Process each view
    for map_2d, camera in zip(maps, cameras, strict=True):
        # Extract first batch element: [N, C, H, W] -> [C, H, W]
        map_single = map_2d[0]  # Assuming batch_size=1
        _fuse_single(
            points=points,
            seg_pred=map_single,
            camera=camera,
            fuse_result=vote_matrix,
        )

    # Final prediction: majority voting
    final_labels = torch.argmax(vote_matrix, dim=1)  # [N] class indices

    # Set ignore_value for points that received no votes from any camera
    no_votes_mask = torch.sum(vote_matrix, dim=1) == 0  # [N] points with zero votes
    final_labels[no_votes_mask] = ignore_value

    return final_labels


def _fuse_single(
    points: torch.Tensor,  # [N, 3] 3D point coordinates
    seg_pred: torch.Tensor,  # [C, H, W] segmentation probability map
    camera: Camera,
    fuse_result: torch.Tensor,  # [N, C] vote accumulation matrix (modified in-place)
) -> None:
    """
    Process a single view and accumulate votes for visible points.

    Args:
        points: [N, 3] 3D point coordinates
        map: [C, H, W] segmentation probability map
        camera: Camera containing intrinsics/extrinsics/convention
        fuse_result: [N, C] vote accumulation matrix (modified in-place)
    """
    # Convert map to points device for consistency
    seg_pred = seg_pred.to(device=points.device)

    H, W = seg_pred.shape[-2:]  # Image dimensions

    # Get visible points and their 2D projections
    visible_points_2d, visible_mask = _get_visible_mask(
        points=points,
        camera=camera,
        image_shape=(H, W),
    )  # visible_points_2d: [num_visible, 2], visible_mask: [N]

    if visible_mask.sum() == 0:
        return  # No visible points in this view

    # Sample segmentation map at visible points
    visible_points_pred = _sample_map(
        coordinates=visible_points_2d, seg_pred=seg_pred
    )  # [num_visible, C]

    # Get hard predictions (class indices)
    visible_points_cls = visible_points_pred.argmax(dim=-1)  # [num_visible]

    # Accumulate votes using advanced indexing
    visible_indices = visible_mask.nonzero(as_tuple=True)[0]  # [num_visible]
    fuse_result[visible_indices, visible_points_cls] += 1
    return


def _get_visible_mask(
    points: torch.Tensor,  # [N, 3] 3D points
    camera: Camera,
    image_shape: Tuple[int, int],  # (H, W) image dimensions
    occlusion_threshold: float = 0.01,  # Depth tolerance for occlusion (meters)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute visibility mask and 2D projections for points with occlusion handling.

    Args:
        points: [N, 3] 3D point coordinates in world space
        camera: Camera containing intrinsics/extrinsics/convention
        image_shape: (H, W) image dimensions
        occlusion_threshold: Depth tolerance for occlusion testing (meters)

    Returns:
        visible_points_2d: [num_visible, 2] 2D coordinates of visible points
        visibility_mask: [N] boolean mask indicating which points are visible
    """
    H, W = image_shape  # Image dimensions

    # Transform points to camera coordinates
    points_cam = _transform_to_camera(
        points=points,
        camera=camera,
    )  # [N, 3]

    # Check if points are in front of camera
    depth_mask = points_cam[:, 2] > 0.01  # [N] - points with positive depth

    # Project all points to image plane
    points_2d_all = _project_to_image(points_cam=points_cam, camera=camera)  # [N, 2]

    # Check if within image bounds
    bounds_mask = (
        (points_2d_all[:, 0] >= 0)
        & (points_2d_all[:, 0] < W)
        & (points_2d_all[:, 1] >= 0)
        & (points_2d_all[:, 1] < H)
    )  # [N]

    # Z-buffer for occlusion handling (following documentation approach)
    depth_buffer = _render_depth_buffer(
        points_cam=points_cam, points_2d=points_2d_all, image_shape=image_shape
    )  # [H, W]
    occlusion_mask = _check_occlusion(
        points_cam=points_cam,
        points_2d=points_2d_all,
        depth_buffer=depth_buffer,
        occlusion_threshold=occlusion_threshold,
    )  # [N]

    # Combine all visibility conditions
    visibility_mask = depth_mask & bounds_mask & ~occlusion_mask  # [N]

    # Extract 2D coordinates of visible points only
    visible_points_2d = points_2d_all[visibility_mask]  # [num_visible, 2]

    return visible_points_2d, visibility_mask


def _transform_to_camera(
    points: torch.Tensor,  # [N, 3] points in world coordinates
    camera: Camera,
) -> torch.Tensor:
    """
    Transform 3D points from world to camera coordinates.

    Args:
        points: [N, 3] 3D points in world coordinates
        camera: Camera describing pose and convention

    Returns:
        points_cam: [N, 3] points in camera coordinates (OpenCV convention)
    """
    # Transform extrinsics to OpenCV convention for consistent processing
    camera = camera.to(convention="opencv")
    world_to_camera = camera.w2c

    N = points.shape[0]  # Number of points
    device = points.device

    # Convert to homogeneous coordinates
    ones = torch.ones(N, 1, device=device)  # [N, 1]
    points_homo = torch.cat([points, ones], dim=1)  # [N, 4]

    # Apply transformation: camera_coords = world_to_camera @ world_coords
    points_cam_homo = (
        world_to_camera @ points_homo.T
    ).T  # [4, 4] @ [4, N] -> [4, N] -> [N, 4]

    # Drop homogeneous coordinate
    points_cam = points_cam_homo[:, :3]  # [N, 3]

    return points_cam


def _project_to_image(
    points_cam: torch.Tensor,  # [N, 3] points in camera coordinates
    camera: Camera,
) -> torch.Tensor:
    """
    Project 3D camera coordinates to 2D image coordinates.

    Args:
        points_cam: [N, 3] points in camera coordinates (x, y, z)
        camera: Camera containing intrinsics

    Returns:
        points_2d: [N, 2] projected 2D coordinates (x, y) in pixels
    """
    N = points_cam.shape[0]  # Number of points
    device = points_cam.device

    # Avoid division by zero
    depths = points_cam[:, 2:3].clamp(min=1e-6)  # [N, 1]

    # Normalize by depth (perspective projection)
    points_normalized = points_cam / depths  # [N, 3]

    # Apply camera intrinsics
    # K @ [x/z, y/z, 1]^T gives [u, v, 1]^T in pixel coordinates
    points_2d_homo = (
        camera.intrinsics @ points_normalized.T
    ).T  # [3, 3] @ [3, N] -> [3, N] -> [N, 3]

    # Extract pixel coordinates
    points_2d = points_2d_homo[:, :2]  # [N, 2]

    return points_2d


def _render_depth_buffer(
    points_cam: torch.Tensor,  # [N, 3] points in camera coordinates
    points_2d: torch.Tensor,  # [N, 2] projected 2D coordinates
    image_shape: Tuple[int, int],  # (H, W) image dimensions
) -> torch.Tensor:
    """
    Create a depth buffer for occlusion testing.

    The depth buffer stores the minimum depth (closest point) at each pixel.
    This is essential for determining which points are visible vs occluded.

    Returns:
        depth_buffer: [H, W] minimum depth at each pixel
    """
    H, W = image_shape
    device = points_cam.device

    # Initialize with maximum depth
    depth_buffer = torch.full(
        (H, W), float('inf'), device=device, dtype=points_cam.dtype
    )

    # Round 2D coordinates to pixel indices
    pixel_x = points_2d[:, 0].long().clamp(0, W - 1)
    pixel_y = points_2d[:, 1].long().clamp(0, H - 1)

    # Get depths (z-coordinates in camera space)
    depths = points_cam[:, 2]  # Positive depths (points are already filtered)

    # Sort by depth (farthest to closest) so closest points overwrite
    sort_indices = torch.argsort(depths, descending=True)
    sorted_pixel_y = pixel_y[sort_indices]
    sorted_pixel_x = pixel_x[sort_indices]
    sorted_depths = depths[sort_indices]

    # Assign depths - closest points write last and become final values
    depth_buffer[sorted_pixel_y, sorted_pixel_x] = sorted_depths

    return depth_buffer.to(torch.float32)


def _check_occlusion(
    points_cam: torch.Tensor,  # [N, 3] points in camera coordinates
    points_2d: torch.Tensor,  # [N, 2] projected 2D coordinates
    depth_buffer: torch.Tensor,  # [H, W] minimum depth at each pixel
    occlusion_threshold: float = 0.01,  # Tolerance for depth comparison (meters)
) -> torch.Tensor:
    """
    Check which points are occluded by comparing with depth buffer.

    A point is considered occluded if its depth is significantly larger than
    the minimum depth stored in the depth buffer at its projected pixel location.

    Args:
        occlusion_threshold: Depth tolerance in meters. Points within this
                           distance of the depth buffer value are considered visible.

    Returns:
        occlusion_mask: [N] boolean mask, True if point is occluded
    """
    H, W = depth_buffer.shape

    # Get depths of all points
    point_depths = points_cam[:, 2]

    # Round 2D coordinates to pixel indices
    pixel_x = points_2d[:, 0].long().clamp(0, W - 1)
    pixel_y = points_2d[:, 1].long().clamp(0, H - 1)

    # Get minimum depth at each point's pixel location
    min_depths_at_pixels = depth_buffer[pixel_y, pixel_x]

    # Point is occluded if its depth is significantly larger than minimum
    occlusion_mask = point_depths > (min_depths_at_pixels + occlusion_threshold)

    return occlusion_mask


def _sample_map(
    coordinates: torch.Tensor,  # [num_visible, 2] 2D coordinates
    seg_pred: torch.Tensor,  # [C, H, W] feature map to sample
) -> torch.Tensor:
    """
    Sample feature map using bilinear interpolation at given 2D coordinates.

    Args:
        coordinates: [num_visible, 2] 2D coordinates (x, y) in pixel space
        feature_map: [C, H, W] feature map to sample from

    Returns:
        sampled_features: [num_visible, C] bilinearly interpolated features
    """
    C, H, W = seg_pred.shape  # Channels, height, width
    N = coordinates.shape[0]  # Number of sample points
    device = seg_pred.device

    if N == 0:
        return torch.zeros(0, C, device=device)

    # Clamp coordinates to valid range
    x = coordinates[:, 0].clamp(0, W - 1)  # [N] x coordinates
    y = coordinates[:, 1].clamp(0, H - 1)  # [N] y coordinates

    # Get integer and fractional parts for bilinear interpolation
    x0 = torch.floor(x).long()  # [N] left x indices
    x1 = torch.clamp(x0 + 1, max=W - 1)  # [N] right x indices
    y0 = torch.floor(y).long()  # [N] top y indices
    y1 = torch.clamp(y0 + 1, max=H - 1)  # [N] bottom y indices

    # Compute interpolation weights
    wx = x - x0.float()  # [N] x interpolation weights
    wy = y - y0.float()  # [N] y interpolation weights

    # Sample corner values from feature map
    # feature_map is [C, H, W], so we index as [channel, y, x]
    f00 = seg_pred[:, y0, x0].T  # [C, N] -> [N, C] top-left
    f01 = seg_pred[:, y0, x1].T  # [C, N] -> [N, C] top-right
    f10 = seg_pred[:, y1, x0].T  # [C, N] -> [N, C] bottom-left
    f11 = seg_pred[:, y1, x1].T  # [C, N] -> [N, C] bottom-right

    # Bilinear interpolation
    wx = wx.unsqueeze(1)  # [N, 1]
    wy = wy.unsqueeze(1)  # [N, 1]

    f0 = f00 * (1 - wx) + f01 * wx  # [N, C] top interpolation
    f1 = f10 * (1 - wx) + f11 * wx  # [N, C] bottom interpolation

    interpolated = f0 * (1 - wy) + f1 * wy  # [N, C] final interpolation

    return interpolated  # [N, C]
