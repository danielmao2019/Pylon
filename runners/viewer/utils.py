"""Shared utilities for the runners viewer."""
from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
import json


def format_value(value: Any) -> str:
    """Format a value for display, handling special cases."""
    try:
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        elif isinstance(value, torch.Tensor):
            return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
        elif isinstance(value, np.ndarray):
            return f"Array(shape={value.shape}, dtype={value.dtype})"
        else:
            return str(value)
    except Exception as e:
        return f"Error formatting value: {str(e)}"


def tensor_to_image(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a tensor to a displayable image array.
    
    Args:
        tensor: Input tensor or array
        
    Returns:
        Normalized numpy array suitable for display
    """
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().numpy()
    else:
        img = tensor

    # Handle normalization with zero division check
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    
    if img.ndim == 2:  # Grayscale image
        return img
    elif img.ndim == 3:  # RGB image (C, H, W) -> (H, W, C)
        if img.shape[0] > 3:  # If more than 3 channels, take first 3
            img = img[:3]
        return np.transpose(img, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported tensor shape: {img.shape}")


def create_image_figure(
    img_array: np.ndarray,
    title: str = "",
    colorscale: str = "Viridis",
    height: int = 400
) -> go.Figure:
    """Create a plotly figure for displaying an image.
    
    Args:
        img_array: Image array to display
        title: Title for the figure
        colorscale: Color scale to use for the image
        height: Height of the figure in pixels
        
    Returns:
        Plotly Figure object
    """
    fig = px.imshow(
        img_array,
        title=title,
        color_continuous_scale=colorscale
    )
    
    fig.update_layout(
        title_x=0.5,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_showscale=True,
        showlegend=False,
        dragmode=False,
        height=height
    )
    
    return fig


def get_image_stats(img: torch.Tensor, change_map: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Get statistical information about an image.

    Args:
        img: Image tensor of shape (C, H, W)
        change_map: Optional tensor with change classes for each pixel

    Returns:
        Dictionary with image statistics
    """
    if not isinstance(img, torch.Tensor):
        return {}

    try:
        # Basic stats
        img_np = img.detach().cpu().numpy()
        stats = {
            "Shape": f"{img_np.shape}",
            "Min Value": f"{img_np.min():.4f}",
            "Max Value": f"{img_np.max():.4f}",
            "Mean Value": f"{img_np.mean():.4f}",
            "Std Dev": f"{img_np.std():.4f}"
        }

        # Add change map statistics if provided
        if change_map is not None:
            if change_map.dim() > 2 and change_map.shape[0] > 1:
                # Multi-class change map
                change_classes = torch.argmax(change_map, dim=0)
                num_classes = change_map.shape[0]
                class_distribution = {
                    i: float((change_classes == i).sum()) / change_classes.numel() * 100
                    for i in range(num_classes)
                }
                stats["Number of Classes"] = num_classes
                stats["Class Distribution"] = {
                    f"Class {i}": f"{pct:.2f}%" 
                    for i, pct in class_distribution.items()
                }
            else:
                # Binary change map
                changes = change_map[0] if change_map.dim() > 2 else change_map
                percent_changed = float((changes > 0.5).sum()) / changes.numel() * 100
                stats["Changed Pixels"] = f"{percent_changed:.2f}%"
                stats["Change Min"] = f"{float(changes.min()):.4f}"
                stats["Change Max"] = f"{float(changes.max()):.4f}"

        return stats

    except Exception as e:
        return {"Error": str(e)}


def get_default_colors():
    """Return default color mapping for visualization."""
    return {
        0: [0, 0, 0],      # Background (black)
        1: [255, 0, 0],    # Change class 1 (red)
        2: [0, 255, 0],    # Change class 2 (green)
        3: [0, 0, 255],    # Change class 3 (blue)
    }
