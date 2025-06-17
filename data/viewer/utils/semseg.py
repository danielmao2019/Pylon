"""Utility functions for semantic segmentation visualization."""
from typing import Dict, List, Optional, Any
import random
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go


def get_color_for_class(class_id: Any) -> str:
    """Generate a deterministic color for any hashable class identifier.

    Args:
        class_id: Any hashable object (int, str, tuple, etc.) representing a class

    Returns:
        Hex color code
    """
    # Get a hash value for the class_id
    # Use abs() to ensure positive values and % 360 to get a hue value
    hue = abs(hash(class_id)) % 360

    # Convert hue to RGB using HSL color space
    # We'll use fixed saturation and lightness for better visibility
    saturation = 0.8
    lightness = 0.5

    # Convert HSL to RGB
    h = hue / 360.0
    s = saturation
    l = lightness

    if s == 0:
        r = g = b = l
    else:
        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p

        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q

        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)

    # Convert to hex
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


def get_colors_for_classes(class_ids: List[Any]) -> List[str]:
    """Generate deterministic colors for a list of class identifiers.

    Args:
        class_ids: List of hashable objects representing classes

    Returns:
        List of hex color codes
    """
    return [get_color_for_class(class_id) for class_id in class_ids]


def tensor_to_semseg(tensor: torch.Tensor, class_ids: Optional[List[Any]] = None) -> np.ndarray:
    """Convert a segmentation tensor to a colored RGB image.

    Args:
        tensor: Can be one of:
            - Semantic segmentation tensor of shape (H, W) or (1, H, W) with class indices
            - List of binary masks of shape (N, H, W) where N is number of instances
        class_ids: Optional list of class identifiers. If None, will use unique values in tensor
            or indices for binary masks.

    Returns:
        Numpy array of shape (H, W, 3) with RGB colors
    """
    if tensor.dim() == 3 and tensor.shape[0] > 1:
        # Handle list of binary masks
        if class_ids is None:
            class_ids = list(range(tensor.shape[0]))
        # Convert to semantic segmentation by taking argmax
        tensor = torch.argmax(tensor, dim=0)
    elif tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    # Get unique classes if not provided
    if class_ids is None:
        class_ids = torch.unique(tensor).tolist()

    # Generate colors for each class
    colors = get_colors_for_classes(class_ids)

    # Create colored segmentation map
    seg_np = tensor.cpu().numpy()
    colored_map = np.zeros((*seg_np.shape, 3), dtype=np.uint8)

    for class_idx, color in zip(class_ids, colors):
        mask = seg_np == class_idx
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        colored_map[mask] = [r, g, b]

    return colored_map


def create_semseg_figure(
    seg_map: torch.Tensor,
    class_ids: Optional[List[Any]] = None,
    title: str = "Segmentation Map"
) -> go.Figure:
    """Create a segmentation map figure.

    Args:
        seg_map: Segmentation tensor (see tensor_to_semseg for supported formats)
        class_ids: Optional list of class identifiers
        title: Figure title
    """
    # Convert segmentation map to RGB
    colored_map = tensor_to_semseg(seg_map, class_ids)

    # Create figure
    fig = px.imshow(
        colored_map,
        title=title
    )

    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )

    return fig


def get_semseg_stats(
    seg_map: torch.Tensor,
    class_ids: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Get statistical information about a segmentation map."""
    if not isinstance(seg_map, torch.Tensor):
        return {}

    # Convert to numpy and get unique classes
    if seg_map.dim() == 3 and seg_map.shape[0] > 1:
        # Handle list of binary masks
        if class_ids is None:
            class_ids = list(range(seg_map.shape[0]))
        seg_np = torch.argmax(seg_map, dim=0).cpu().numpy()
    else:
        seg_np = tensor_to_semseg(seg_map, class_ids)[..., 0]  # Just use one channel for stats
        if class_ids is None:
            class_ids = torch.unique(seg_map).tolist()

    stats = {
        "Shape": f"{seg_np.shape}",
        "Number of Classes": len(class_ids),
        "Class Distribution": {}
    }

    # Calculate class distribution
    for class_idx in class_ids:
        class_pixels = (seg_np == class_idx).sum()
        class_percentage = (class_pixels / seg_np.size) * 100
        stats["Class Distribution"][str(class_idx)] = f"{class_percentage:.2f}%"

    return stats
