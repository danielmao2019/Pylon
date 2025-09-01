"""Segmentation rendering from point clouds using projection."""

import torch
from typing import Dict, Tuple, Union
from utils.point_cloud_ops.rendering.render_common import (
    validate_rendering_inputs,
    prepare_points_for_rendering,
    create_valid_mask
)


def render_segmentation_from_pointcloud(
    pc_data: Dict[str, torch.Tensor],
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = "opengl",
    ignore_value: int = 255,
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
        ignore_value: Fill value for pixels with no point projections (default: 255)
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
    # Segmentation-specific validation
    assert key in pc_data, f"pc_data must contain '{key}' key, got keys: {list(pc_data.keys())}"

    # Common input validation
    validate_rendering_inputs(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        convention=convention,
        ignore_value=ignore_value,
        return_mask=return_mask
    )

    labels = pc_data[key]
    assert labels.numel() > 0, f"Labels tensor must not be empty, got {labels.numel()} elements"

    render_width, render_height = resolution

    # Use common preprocessing
    points, filtered_indices = prepare_points_for_rendering(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        convention=convention
    )

    # Filter and sort labels to match processed points
    labels = labels[filtered_indices]

    # Step 10: Allocate segmentation map
    seg_map = torch.full(
        (render_height, render_width),
        ignore_value,
        dtype=labels.dtype,
        device=labels.device
    )

    # Step 11: Fill segmentation map with sorted labels
    seg_map[points[:, 1].long(), points[:, 0].long()] = labels

    # Convert to int64 for final output
    seg_map = seg_map.to(dtype=torch.int64)

    if return_mask:
        # Step 12: Create valid mask
        valid_mask = create_valid_mask(points, resolution, points.device)
        return seg_map, valid_mask
    else:
        return seg_map
