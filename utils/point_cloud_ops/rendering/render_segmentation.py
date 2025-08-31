"""Segmentation rendering from point clouds using projection."""

import torch
from typing import Dict, Tuple, Union
from .render_common import prepare_points_for_rendering


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
    # Segmentation-specific validation
    assert key in pc_data, f"pc_data must contain '{key}' key, got keys: {list(pc_data.keys())}"
    labels = pc_data[key]
    assert isinstance(labels, torch.Tensor), f"labels must be torch.Tensor, got {type(labels)}"
    assert labels.ndim == 1, f"labels must be 1D tensor with shape (N,), got shape {labels.shape}"
    assert labels.shape[0] == pc_data['pos'].shape[0], f"labels length {labels.shape[0]} != points length {pc_data['pos'].shape[0]}"
    assert labels.device == pc_data['pos'].device, f"points device {pc_data['pos'].device} != labels device {labels.device}"
    assert isinstance(ignore_index, int), f"ignore_index must be int, got {type(ignore_index)}"
    assert 0 <= ignore_index <= 255, f"ignore_index must be in range [0, 255], got {ignore_index}"
    assert isinstance(key, str), f"key must be str, got {type(key)}"
    assert isinstance(return_mask, bool), f"return_mask must be bool, got {type(return_mask)}"

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
        ignore_index,
        dtype=labels.dtype,
        device=labels.device
    )

    # Step 11: Fill segmentation map with sorted labels
    seg_map[points[:, 1].long(), points[:, 0].long()] = labels

    # Convert to int64 for final output
    seg_map = seg_map.to(dtype=torch.int64)

    if return_mask:
        # Step 12: Create valid mask
        valid_mask = torch.zeros(
            (render_height, render_width),
            dtype=torch.bool,
            device=points.device
        )
        valid_mask[points[:, 1].long(), points[:, 0].long()] = True
        
        return seg_map, valid_mask
    else:
        return seg_map
