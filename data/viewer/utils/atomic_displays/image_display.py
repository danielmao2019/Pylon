"""Image display utilities for RGB/grayscale image visualization."""

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch


def create_image_display(
    image: torch.Tensor,
    title: str,
    colorscale: str = "Viridis",
    resolution: Optional[Tuple[int, int]] = None,
    **kwargs: Any,
) -> go.Figure:
    """Create image display for RGB or grayscale images.

    Args:
        image: Image tensor of shape [C, H, W] or [N, C, H, W] (batched)
        title: Title for the image display
        colorscale: Color scale to use for the image
        resolution: Optional display resolution as (height, width) tuple
        **kwargs: Additional arguments

    Returns:
        Plotly figure for image visualization

    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validations
    assert isinstance(image, torch.Tensor), f"{type(image)=}"
    assert image.ndim in [3, 4], f"{image.shape=}"
    assert image.numel() > 0, f"{image.shape=}"
    assert isinstance(title, str), f"{type(title)=}"
    assert isinstance(colorscale, str), f"{type(colorscale)=}"
    assert resolution is None or isinstance(resolution, tuple), f"{type(resolution)=}"
    assert resolution is None or len(resolution) == 2, f"{resolution=}"
    assert resolution is None or all(isinstance(x, int) for x in resolution), (
        f"{resolution=}"
    )
    assert resolution is None or all(x > 0 for x in resolution), f"{resolution=}"
    assert image.ndim != 4 or image.shape[0] == 1, f"{image.shape=}"
    assert (image.ndim == 3 and image.shape[0] >= 1) or (
        image.ndim == 4 and image.shape[1] >= 1
    ), f"{image.shape=}"

    # Input normalizations

    # Handle batched input - extract single sample for visualization
    if image.ndim == 4:
        image = image[0]  # [N, C, H, W] -> [C, H, W]

    # Convert image to numpy for visualization
    img: np.ndarray = _image_to_numpy(image)

    fig = px.imshow(img, title=title, color_continuous_scale=colorscale)

    # Define common layout parameters
    layout_params = {
        'title_x': 0.5,
        'coloraxis_showscale': True,
        'showlegend': False,
        'xaxis': dict(scaleanchor="y", scaleratio=1),  # Lock aspect ratio
        'yaxis': dict(autorange='reversed'),  # Standard image convention
    }

    # Update dimensions if resolution is provided
    if resolution is not None:
        height, width = resolution
        layout_params['height'] = height
        layout_params['width'] = width

    # Apply all layout parameters at once
    fig.update_layout(**layout_params)

    return fig


def _image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a displayable image.

    Args:
        image: Image tensor of shape [C, H, W] where C is 1 (grayscale) or 3+ (RGB/multi-channel)
               2-channel images are not supported as they are uncommon in computer vision

    Returns:
        Numpy array suitable for display

    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validations
    assert isinstance(image, torch.Tensor), f"{type(image)=}"
    assert image.ndim == 3, f"{image.shape=}"
    assert image.numel() > 0, f"{image.shape=}"
    assert image.shape[0] != 2, f"{image.shape=}"

    img: np.ndarray = image.detach().cpu().numpy()

    # Smart normalization: only normalize if image is not already in [0,1] range
    img_min = img.min()
    img_max = img.max()

    # Check if image is already in reasonable [0,1] range
    if img_min >= 0.0 and img_max <= 1.0:
        # Image is already in [0,1] range - no normalization needed
        pass
    elif img_max > img_min:
        # Image needs normalization - stretch to [0,1] range
        img = (img - img_min) / (img_max - img_min)
    else:
        # If all values are the same, return zeros
        img = np.zeros_like(img)

    if img.ndim == 2:  # Already 2D grayscale
        return img
    elif img.ndim == 3:  # Multi-channel image (C, H, W)
        if img.shape[0] == 1:  # Single channel grayscale -> (H, W)
            return img.squeeze(0)  # Remove channel dimension
        elif img.shape[0] == 2:  # 2-channel images not supported
            assert False, (
                "2-channel images are not supported. "
                f"Use 1 channel or 3+ channels. {img.shape=}"
            )
        elif img.shape[0] == 3:  # RGB image -> (H, W, C)
            return np.transpose(img, (1, 2, 0))
        else:  # Multi-channel > 3 -> sample 3 channels -> (H, W, C)
            img = img[random.sample(range(img.shape[0]), 3), :, :]
            return np.transpose(img, (1, 2, 0))
    else:
        assert False, f"Unsupported tensor shape for image conversion: {img.shape}"


def get_image_display_stats(image: torch.Tensor) -> Dict[str, Any]:
    """Get image statistics for display.

    Args:
        image: Image tensor of shape [C, H, W] or [N, C, H, W] (batched)

    Returns:
        Dictionary containing image statistics

    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validations
    assert isinstance(image, torch.Tensor), f"{type(image)=}"
    assert image.ndim in [3, 4], f"{image.shape=}"
    assert image.numel() > 0, f"{image.shape=}"
    assert image.ndim != 4 or image.shape[0] == 1, f"{image.shape=}"

    # Input normalizations
    if image.ndim == 4:
        image = image[0]  # [N, C, H, W] -> [C, H, W]

    # Basic stats
    img_np: np.ndarray = image.detach().cpu().numpy()
    stats: Dict[str, Any] = {
        "Shape": f"{img_np.shape}",
        "Min Value": f"{img_np.min():.4f}",
        "Max Value": f"{img_np.max():.4f}",
        "Mean Value": f"{img_np.mean():.4f}",
        "Std Dev": f"{img_np.std():.4f}",
    }

    return stats
