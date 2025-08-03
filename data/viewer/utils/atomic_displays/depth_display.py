"""Depth display utilities for depth map visualization."""
from typing import Dict, Any, Optional
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def create_depth_display(
    depth: torch.Tensor,
    title: str,
    colorscale: str = 'Viridis',
    **kwargs: Any
) -> go.Figure:
    """Create depth map display with proper visualization.
    
    Args:
        depth: Depth tensor of shape [H, W] with depth values
        title: Title for the depth display
        colorscale: Plotly colorscale for depth visualization
        **kwargs: Additional arguments
        
    Returns:
        Plotly figure for depth visualization
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # CRITICAL: Input validation with fail-fast assertions
    assert isinstance(depth, torch.Tensor), f"Expected torch.Tensor, got {type(depth)}"
    assert depth.ndim == 2, f"Expected 2D tensor [H,W], got shape {depth.shape}"
    assert depth.numel() > 0, f"Depth tensor cannot be empty"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    assert isinstance(colorscale, str), f"Expected str colorscale, got {type(colorscale)}"
    
    # Convert to numpy for visualization
    depth_np = depth.detach().cpu().numpy()
    
    # Create depth visualization
    fig = px.imshow(
        depth_np,
        color_continuous_scale=colorscale,
        title=title,
        labels={'color': 'Depth'}
    )
    
    # Update layout for better visualization
    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        coloraxis_colorbar=dict(title="Depth")
    )
    
    return fig


def get_depth_display_stats(depth: torch.Tensor) -> Dict[str, Any]:
    """Get depth statistics for display.
    
    Args:
        depth: Depth tensor of shape [H, W]
        
    Returns:
        Dictionary containing depth statistics
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validation
    assert isinstance(depth, torch.Tensor), f"Expected torch.Tensor, got {type(depth)}"
    assert depth.ndim == 2, f"Expected 2D tensor [H,W], got shape {depth.shape}"
    assert depth.numel() > 0, f"Depth tensor cannot be empty"
    
    # Calculate statistics
    valid_mask = torch.isfinite(depth) & (depth > 0)
    valid_depth = depth[valid_mask]
    
    if len(valid_depth) == 0:
        return {
            'shape': list(depth.shape),
            'dtype': str(depth.dtype),
            'valid_pixels': 0,
            'min_depth': 'N/A',
            'max_depth': 'N/A',
            'mean_depth': 'N/A',
            'std_depth': 'N/A'
        }
    
    return {
        'shape': list(depth.shape),
        'dtype': str(depth.dtype),
        'valid_pixels': len(valid_depth),
        'total_pixels': depth.numel(),
        'min_depth': float(valid_depth.min()),
        'max_depth': float(valid_depth.max()),
        'mean_depth': float(valid_depth.mean()),
        'std_depth': float(valid_depth.std())
    }