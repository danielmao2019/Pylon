"""Image display utilities for RGB/grayscale image visualization."""
from typing import Dict, Any, Optional
import torch
import plotly.graph_objects as go
from data.viewer.utils.image import create_image_figure, get_image_stats


def create_image_display(
    image: torch.Tensor,
    title: str,
    **kwargs: Any
) -> go.Figure:
    """Create image display for RGB or grayscale images.
    
    This function consolidates RGB and grayscale image display functionality,
    leveraging the existing create_image_figure implementation with fail-fast
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
    
    # Use existing create_image_figure implementation
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
    
    # Use existing get_image_stats implementation
    return get_image_stats(image)