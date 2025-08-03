"""Image display utilities for RGB/grayscale image visualization."""
from typing import Dict, Any, Optional
import random
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go


def image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a displayable image.
    
    Args:
        image: Image tensor to convert
        
    Returns:
        Numpy array suitable for display
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor, got {type(image)}"
    
    if image.ndim == 4:
        assert image.shape[0] == 1, f"Expected batch size 1, got shape {image.shape}"
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
        raise ValueError(f"Unsupported tensor shape for image conversion: {img.shape}")


def create_image_figure(image: torch.Tensor, title: str = "Image", colorscale: str = "Viridis") -> go.Figure:
    """Create a 2D image figure with standard formatting.

    Args:
        image: Image tensor to display
        title: Title for the figure
        colorscale: Color scale to use for the image

    Returns:
        Plotly Figure object
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor, got {type(image)}"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    assert isinstance(colorscale, str), f"Expected str colorscale, got {type(colorscale)}"
    
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
        image: Image tensor of shape (C, H, W)
        change_map: Optional tensor with change classes for each pixel

    Returns:
        Dictionary with image statistics
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor, got {type(image)}"
    
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
        assert isinstance(change_map, torch.Tensor), f"change_map must be torch.Tensor, got {type(change_map)}"
        
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


def create_image_display(
    image: torch.Tensor,
    title: str,
    **kwargs: Any
) -> go.Figure:
    """Create image display for RGB or grayscale images.
    
    This function consolidates RGB and grayscale image display functionality,
    leveraging the create_image_figure implementation with fail-fast
    input validation.
    
    Args:
        image: Image tensor of shape [C, H, W] where C is 1 (grayscale) or 3 (RGB)
        title: Title for the image display
        **kwargs: Additional arguments passed to create_image_figure
        
    Returns:
        Plotly figure for image visualization
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # CRITICAL: Input validation with fail-fast assertions
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor, got {type(image)}"
    assert image.ndim == 3, f"Expected 3D tensor [C,H,W], got shape {image.shape}"
    assert image.shape[0] in [1, 3], f"Expected 1 or 3 channels, got {image.shape[0]}"
    assert image.numel() > 0, f"Image tensor cannot be empty"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    
    # Use create_image_figure implementation
    return create_image_figure(image=image, title=title)


def get_image_display_stats(image: torch.Tensor) -> Dict[str, Any]:
    """Get image statistics for display.
    
    Args:
        image: Image tensor of shape [C, H, W]
        
    Returns:
        Dictionary containing image statistics
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validation
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor, got {type(image)}"
    assert image.ndim == 3, f"Expected 3D tensor [C,H,W], got shape {image.shape}"
    
    # Use get_image_stats implementation
    return get_image_stats(image)