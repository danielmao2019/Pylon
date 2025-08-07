"""Normal display utilities for surface normal visualization."""
from typing import Dict, Any, Optional
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def create_normal_display(
    normals: torch.Tensor,
    title: str,
    **kwargs: Any
) -> go.Figure:
    """Create surface normal display with proper visualization.
    
    Surface normals are typically stored as 3-channel images where each pixel
    contains the (x, y, z) components of the surface normal vector. This function
    visualizes them by converting the normal vectors to RGB colors.
    
    Args:
        normals: Normal tensor of shape [3, H, W] or [N, 3, H, W] (batched) with normal vectors
        title: Title for the normal display
        **kwargs: Additional arguments
        
    Returns:
        Plotly figure for normal visualization
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # CRITICAL: Input validation with fail-fast assertions
    assert isinstance(normals, torch.Tensor), f"Expected torch.Tensor, got {type(normals)}"
    assert normals.ndim in [3, 4], f"Expected 3D [3,H,W] or 4D [N,3,H,W] tensor, got shape {normals.shape}"
    assert normals.numel() > 0, f"Normal tensor cannot be empty"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    
    # Handle batched input - extract single sample for visualization
    if normals.ndim == 4:
        assert normals.shape[0] == 1, f"Expected batch size 1 for visualization, got {normals.shape[0]}"
        normals = normals[0]  # [N, 3, H, W] -> [3, H, W]
    
    # Validate unbatched tensor shape
    assert normals.shape[0] == 3, f"Expected 3 channels for normals, got {normals.shape[0]}"
    
    # Convert normals to RGB visualization
    # Normals are typically in range [-1, 1], we map to [0, 1] for RGB
    normals_normalized = (normals + 1.0) / 2.0
    normals_normalized = torch.clamp(normals_normalized, 0.0, 1.0)
    
    # Convert to numpy and transpose to [H, W, 3] for visualization
    normals_np = normals_normalized.detach().cpu().numpy()
    normals_rgb = np.transpose(normals_np, (1, 2, 0))  # [3, H, W] -> [H, W, 3]
    
    # Create RGB visualization of normals
    fig = px.imshow(
        normals_rgb,
        title=title
    )
    
    # Update layout for better visualization
    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    
    return fig


def get_normal_display_stats(normals: torch.Tensor) -> Dict[str, Any]:
    """Get surface normal statistics for display.
    
    Args:
        normals: Normal tensor of shape [3, H, W] or [N, 3, H, W] (batched)
        
    Returns:
        Dictionary containing normal statistics
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validation
    assert isinstance(normals, torch.Tensor), f"Expected torch.Tensor, got {type(normals)}"
    assert normals.ndim in [3, 4], f"Expected 3D [3,H,W] or 4D [N,3,H,W] tensor, got shape {normals.shape}"
    assert normals.numel() > 0, f"Normal tensor cannot be empty"
    
    # Handle batched input - extract single sample for analysis
    if normals.ndim == 4:
        assert normals.shape[0] == 1, f"Expected batch size 1 for analysis, got {normals.shape[0]}"
        normals = normals[0]  # [N, 3, H, W] -> [3, H, W]
    
    # Validate unbatched tensor shape
    assert normals.shape[0] == 3, f"Expected 3 channels, got {normals.shape[0]}"
    
    # Calculate statistics
    valid_mask = torch.isfinite(normals).all(dim=0)
    valid_normals = normals[:, valid_mask]
    
    if valid_normals.numel() == 0:
        return {
            'shape': list(normals.shape),
            'dtype': str(normals.dtype),
            'valid_pixels': 0,
            'total_pixels': normals.shape[1] * normals.shape[2],
            'x_range': 'N/A',
            'y_range': 'N/A', 
            'z_range': 'N/A',
            'mean_magnitude': 'N/A'
        }
    
    # Calculate normal vector magnitudes
    magnitudes = torch.norm(valid_normals, dim=0)
    
    return {
        'shape': list(normals.shape),
        'dtype': str(normals.dtype),
        'valid_pixels': valid_normals.shape[1],
        'total_pixels': normals.shape[1] * normals.shape[2],
        'x_range': f"[{float(valid_normals[0].min()):.3f}, {float(valid_normals[0].max()):.3f}]",
        'y_range': f"[{float(valid_normals[1].min()):.3f}, {float(valid_normals[1].max()):.3f}]",
        'z_range': f"[{float(valid_normals[2].min()):.3f}, {float(valid_normals[2].max()):.3f}]",
        'mean_magnitude': float(magnitudes.mean()),
        'std_magnitude': float(magnitudes.std())
    }
