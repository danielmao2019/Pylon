"""RGB rendering from point clouds using projection methods."""

import torch
from typing import Dict, Tuple, Union
from .render_common import prepare_points_for_rendering


def render_rgb_from_pointcloud(
    pc_data: Dict[str, torch.Tensor],
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = "opengl",
    ignore_value: float = 0.0,
    return_mask: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render RGB image from point cloud using camera projection.

    Projects 3D point cloud coordinates with RGB colors onto 2D image plane
    using camera parameters and generates an RGB image. Uses efficient tensor
    operations for projection and depth sorting.

    Args:
        pc_data: Point cloud dictionary containing 'pos' key with 3D coordinates
                 and 'rgb' key with color information (mandatory)
        camera_intrinsics: 3x3 camera intrinsics matrix with focal lengths and principal point
        camera_extrinsics: 4x4 camera extrinsics matrix (camera-to-world transform)
        resolution: Target resolution as (width, height) tuple - intrinsics scaled automatically
        convention: Camera extrinsics convention ("opengl" supported, "standard" not implemented)
        ignore_value: Fill value for pixels with no point projections (default: 0.0)
        return_mask: If True, also return valid pixel mask (default: False)

    Returns:
        If return_mask is False:
            RGB image tensor of shape [3, H, W] with normalized values in [0, 1]
        If return_mask is True:
            Tuple of (RGB image tensor, valid mask tensor of shape [H, W] with boolean values)

    Raises:
        AssertionError: If point cloud is empty, RGB data is missing, or no points project within bounds
        NotImplementedError: If convention other than "opengl" is specified
    """
    # RGB-specific validation
    assert 'rgb' in pc_data, f"pc_data must contain 'rgb' key, got keys: {list(pc_data.keys())}"
    assert isinstance(ignore_value, (int, float)), f"ignore_value must be int or float, got {type(ignore_value)}"
    assert isinstance(return_mask, bool), f"return_mask must be bool, got {type(return_mask)}"

    render_width, render_height = resolution
    colors = pc_data['rgb']
    assert colors.numel() > 0, f"Colors tensor must not be empty, got {colors.numel()} elements"

    # Handle color normalization
    colors = colors.clone().to(dtype=torch.float64)
    integer_dtypes = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    is_integer_dtype = colors.dtype in integer_dtypes
    is_in_255_range = colors.min() >= 0 and colors.max() <= 255 and colors.max() > 1.0

    # Normalize colors if needed
    if is_integer_dtype or is_in_255_range:
        colors = colors / 255.0

    # Ensure colors are in [0, 1] range
    colors = torch.clamp(colors, 0.0, 1.0)

    # Use common preprocessing
    points, filtered_indices = prepare_points_for_rendering(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        convention=convention
    )

    # Filter and sort colors to match processed points
    colors = colors[filtered_indices]

    # Step 11: Allocate RGB image (3 channels)
    rgb_image = torch.full(
        (3, render_height, render_width),
        ignore_value,
        dtype=torch.float32,
        device=points.device
    )

    # Step 12: Fill RGB image with sorted colors (closest points overwrite farther ones)
    # Assign all RGB channels at once using advanced indexing
    rgb_image[:, points[:, 1].long(), points[:, 0].long()] = colors.float().T

    if return_mask:
        # Step 13: Create valid mask
        valid_mask = torch.zeros(
            (render_height, render_width),
            dtype=torch.bool,
            device=points.device
        )
        valid_mask[points[:, 1].long(), points[:, 0].long()] = True
        
        return rgb_image, valid_mask
    else:
        return rgb_image
