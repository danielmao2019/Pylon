"""Utility functions for semantic segmentation visualization."""
from typing import Dict, Union, Any
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go


def get_color(idx: Any) -> str:
    """Generate a deterministic color for any hashable class identifier.

    Args:
        idx: Any hashable object (int, str, tuple, etc.)

    Returns:
        Hex color code
    """
    # Convert non-integer indices to integers using hash
    if not isinstance(idx, int):
        idx = abs(hash(idx))

    # Use golden ratio to get well-distributed hues
    # This ensures colors are visually distinct even for consecutive indices
    golden_ratio = 0.618033988749895
    hue = (idx * golden_ratio) % 1.0

    # Use high saturation and value for better visibility
    saturation = 0.8
    lightness = 0.6  # Increased from 0.5 for better visibility

    # Convert HSL to RGB
    h = hue
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


def tensor_to_semseg(seg: Union[torch.Tensor, Dict[str, Any]]) -> np.ndarray:
    """Convert a segmentation representation to a colored RGB image.

    Args:
        seg: Can be one of:
            - 2D tensor of shape (H, W) with class indices
            - Dict with keys:
                - "masks": List[torch.Tensor] of binary masks
                - "indices": List[Any] of corresponding indices

    Returns:
        Numpy array of shape (H, W, 3) with RGB colors
    """
    if isinstance(seg, dict):
        # Handle dict format with masks and indices
        masks = seg["masks"]
        indices = seg["indices"]

        # Stack masks and take argmax
        stacked_masks = torch.stack(masks)
        tensor = torch.argmax(stacked_masks, dim=0)
    else:
        # Handle tensor format
        if seg.dim() == 3:
            seg = seg.squeeze(0)
        tensor = seg
        indices = torch.unique(tensor).tolist()

    # Generate colors for each index
    colors = [get_color(idx) for idx in indices]

    # Create colored segmentation map
    seg_np = tensor.cpu().numpy()
    colored_map = np.zeros((*seg_np.shape, 3), dtype=np.uint8)

    for idx, color in zip(indices, colors):
        mask = seg_np == idx
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        colored_map[mask] = [r, g, b]

    return colored_map


def create_semseg_figure(
    seg: Union[torch.Tensor, Dict[str, Any]],
    title: str = "Segmentation Map",
) -> go.Figure:
    """Create a segmentation map figure.

    Args:
        seg: Segmentation representation (see tensor_to_semseg for supported formats)
        title: Figure title
    """
    # Convert segmentation map to RGB
    colored_map = tensor_to_semseg(seg)

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
    seg: Union[torch.Tensor, Dict[str, Any]]
) -> Dict[str, Any]:
    """Get statistical information about a segmentation map.

    Args:
        seg: Segmentation representation (see tensor_to_semseg for supported formats)
    """
    if isinstance(seg, dict):
        # Handle dict format with masks and indices
        masks = seg["masks"]
        indices = seg["indices"]
        # Stack masks and take argmax
        stacked_masks = torch.stack(masks)
        seg_np = torch.argmax(stacked_masks, dim=0).cpu().numpy()
    else:
        # Handle tensor format
        if seg.dim() == 3:
            seg = seg.squeeze(0)
        seg_np = seg.cpu().numpy()
        indices = torch.unique(seg).tolist()

    stats = {
        "Shape": f"{seg_np.shape}",
        "Number of Classes": len(indices),
        "Class Distribution": {}
    }

    # Calculate class distribution
    for idx in indices:
        class_pixels = (seg_np == idx).sum()
        class_percentage = (class_pixels / seg_np.size) * 100
        stats["Class Distribution"][str(idx)] = f"{class_percentage:.2f}%"

    return stats
