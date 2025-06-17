"""Utility functions for image visualization."""
from typing import Dict, Any, Optional
import random
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go


def image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a displayable image."""
    assert isinstance(image, torch.Tensor), f"{image=}"
    if image.ndim == 4:
        assert image.shape[0] == 1, f"{image.shape=}"
        image = image.squeeze(0)
    if image.ndim == 3:
        image = image.squeeze(0)

    img: np.ndarray = image.cpu().numpy()

    # Handle normalization with zero division check
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        # If all values are the same, return zeros
        img = np.zeros_like(img)

    if img.ndim == 2:  # Grayscale image
        return img
    elif img.ndim == 3:  # RGB image (C, H, W) -> (H, W, C)
        if img.shape[0] > 3:
            img = img[random.sample(range(img.shape[0]), 3), :, :]
        return np.transpose(img, (1, 2, 0))
    else:
        raise ValueError("Unsupported tensor shape for image conversion")


def create_image_figure(image: torch.Tensor, title: str = "Image", colorscale: str = "Viridis") -> go.Figure:
    """Create a 2D image figure with standard formatting.

    Args:
        tensor: Image tensor to display
        title: Title for the figure
        colorscale: Color scale to use for the image

    Returns:
        Plotly Figure object
    """
    img: np.ndarray = image_to_numpy(image)

    fig = px.imshow(
        img,
        title=title,
        color_continuous_scale=colorscale
    )

    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=True,
        showlegend=False,
        height=400
    )

    return fig


def get_image_stats(image: torch.Tensor, change_map: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Get statistical information about a 2D image.

    Args:
        img: Image tensor of shape (C, H, W)
        change_map: Optional tensor with change classes for each pixel

    Returns:
        Dictionary with image statistics
    """
    if not isinstance(image, torch.Tensor):
        return {}

    # Basic stats
    img_np: np.ndarray = image.detach().cpu().numpy()
    stats: Dict[str, Any] = {
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
            change_classes: torch.Tensor = torch.argmax(change_map, dim=0)
            num_classes: int = change_map.shape[0]
            class_distribution: Dict[int, float] = {
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
            changes: torch.Tensor = change_map[0] if change_map.dim() > 2 else change_map
            percent_changed: float = float((changes > 0.5).sum()) / changes.numel() * 100
            stats["Changed Pixels"] = f"{percent_changed:.2f}%"
            stats["Change Min"] = f"{float(changes.min()):.4f}"
            stats["Change Max"] = f"{float(changes.max()):.4f}"

    return stats
