"""Utility functions for semantic segmentation visualization."""
from typing import Dict, List, Optional, Any
import random
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from .image import tensor_to_image


def generate_unique_colors(num_classes: int) -> List[str]:
    """Generate a list of unique colors for class visualization.

    Args:
        num_classes: Number of classes to generate colors for

    Returns:
        List of hex color codes
    """
    # Use a fixed seed for reproducibility
    random.seed(42)

    # Generate colors using HSV color space for better distribution
    colors = []
    for i in range(num_classes):
        # Use golden ratio to get well-distributed hues
        hue = (i * 0.618033988749895) % 1.0
        # Use high saturation and value for better visibility
        saturation = 0.8
        value = 0.9

        # Convert HSV to RGB
        h = hue
        s = saturation
        v = value

        if s == 0.0:
            r, g, b = v, v, v
        else:
            h *= 6.0
            i = int(h)
            f = h - i
            p = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))

            if i == 0:
                r, g, b = v, t, p
            elif i == 1:
                r, g, b = q, v, p
            elif i == 2:
                r, g, b = p, v, t
            elif i == 3:
                r, g, b = p, q, v
            elif i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q

        # Convert to hex
        hex_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        colors.append(hex_color)

    return colors


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
