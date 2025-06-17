"""Utility functions for semantic segmentation visualization."""
from typing import Dict, List, Optional, Any
import random
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from .image import tensor_to_image


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


def tensor_to_semseg(tensor: torch.Tensor) -> np.ndarray:
    """Convert a semantic segmentation tensor to a displayable image.
    
    Args:
        tensor: Semantic segmentation tensor of shape (H, W) or (1, H, W)
        
    Returns:
        Numpy array of shape (H, W) with class indices
    """
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    return tensor.cpu().numpy()


def create_semseg_figure(
    seg_map: torch.Tensor,
    colors: List[str],
    class_names: Optional[Dict[int, str]] = None,
    title: str = "Segmentation Map"
) -> go.Figure:
    """Create a segmentation map figure with class colors."""
    # Convert segmentation map to numpy
    seg_np = tensor_to_semseg(seg_map)

    # Create a colored segmentation map
    colored_map = np.zeros((*seg_np.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(colors):
        mask = seg_np == class_idx
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        colored_map[mask] = [r, g, b]

    fig = px.imshow(
        colored_map,
        title=title
    )

    # Add legend for classes
    if class_names:
        for class_idx, class_name in class_names.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=colors[class_idx]),
                name=class_name,
                showlegend=True
            ))

    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        height=400
    )

    return fig


def get_semseg_stats(
    seg_map: torch.Tensor,
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """Get statistical information about a segmentation map."""
    if not isinstance(seg_map, torch.Tensor):
        return {}

    # Basic stats
    seg_np = tensor_to_semseg(seg_map)
    num_classes = int(seg_map.max().item()) + 1

    stats = {
        "Shape": f"{seg_np.shape}",
        "Number of Classes": num_classes,
        "Class Distribution": {}
    }

    # Calculate class distribution
    for class_idx in range(num_classes):
        class_name = class_names.get(class_idx, f"Class {class_idx}") if class_names else f"Class {class_idx}"
        class_pixels = (seg_np == class_idx).sum()
        class_percentage = (class_pixels / seg_np.size) * 100
        stats["Class Distribution"][class_name] = f"{class_percentage:.2f}%"

    return stats
