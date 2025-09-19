"""Normal display utilities for surface normal visualization."""
from typing import Dict, Any
import torch
import plotly.graph_objects as go
from data.viewer.utils.atomic_displays.image_display import create_image_display


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
    
    # Use existing image display utility
    fig = create_image_display(
        image=normals_normalized,
        title=title,
        **kwargs
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
    total_pixels = normals.shape[1] * normals.shape[2]
    
    # Create masks for different pixel types
    nan_mask = torch.isnan(normals).any(dim=0)
    inf_mask = torch.isinf(normals).any(dim=0)
    ignore_mask = torch.norm(normals, dim=0) <= 1e-8
    valid_mask = ~(nan_mask | inf_mask | ignore_mask)
    
    # Count pixels
    nan_pixels = nan_mask.sum().item()
    inf_pixels = inf_mask.sum().item()
    ignore_pixels = ignore_mask.sum().item()
    valid_pixels = valid_mask.sum().item()
    
    # Calculate percentages
    nan_pct = (nan_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    inf_pct = (inf_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    ignore_pct = (ignore_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    valid_pct = (valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    valid_normals = normals[:, valid_mask]
    
    if valid_normals.numel() == 0:
        return {
            'shape': list(normals.shape),
            'dtype': str(normals.dtype),
            'total_pixels': total_pixels,
            'valid_pixels': f"{valid_pixels} ({valid_pct:.1f}%)",
            'nan_pixels': f"{nan_pixels} ({nan_pct:.1f}%)",
            'inf_pixels': f"{inf_pixels} ({inf_pct:.1f}%)",
            'ignore_pixels': f"{ignore_pixels} ({ignore_pct:.1f}%)",
            'x_range': 'N/A',
            'y_range': 'N/A', 
            'z_range': 'N/A',
            'mean_magnitude': 'N/A'
        }
    
    return {
        'shape': list(normals.shape),
        'dtype': str(normals.dtype),
        'total_pixels': total_pixels,
        'valid_pixels': f"{valid_pixels} ({valid_pct:.1f}%)",
        'nan_pixels': f"{nan_pixels} ({nan_pct:.1f}%)",
        'inf_pixels': f"{inf_pixels} ({inf_pct:.1f}%)",
        'ignore_pixels': f"{ignore_pixels} ({ignore_pct:.1f}%)",
        'x_range': f"[{float(valid_normals[0].min()):.3f}, {float(valid_normals[0].max()):.3f}]",
        'y_range': f"[{float(valid_normals[1].min()):.3f}, {float(valid_normals[1].max()):.3f}]",
        'z_range': f"[{float(valid_normals[2].min()):.3f}, {float(valid_normals[2].max()):.3f}]",
        'mean_magnitude': float(torch.norm(valid_normals, dim=0).mean()),
        'std_magnitude': float(torch.norm(valid_normals, dim=0).std())
    }
