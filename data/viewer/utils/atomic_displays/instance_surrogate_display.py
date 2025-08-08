"""Instance segmentation surrogate display utilities.

This module handles visualization of instance segmentation in surrogate representation,
where each pixel contains x, y coordinate offsets to the instance centroid.
This format is commonly used in instance segmentation research.
"""
from typing import Dict, Any
import torch
import plotly.graph_objects as go
from data.viewer.utils.atomic_displays.segmentation_display import create_segmentation_display


def create_instance_surrogate_display(
    instance_surrogate: torch.Tensor,
    title: str,
    ignore_value: int = 250,
    **kwargs: Any
) -> go.Figure:
    """Create instance segmentation display from coordinate surrogate representation.
    
    This function handles the common instance segmentation surrogate format where
    each pixel contains Y and X coordinate offsets relative to the instance centroid.
    This representation is used in many instance segmentation papers and datasets.
    
    Args:
        instance_surrogate: Tensor of shape [2, H, W] or [N, 2, H, W] (batched) with coordinate offsets
            - Channel 0: Y-offset to instance centroid  
            - Channel 1: X-offset to instance centroid
        title: Title for the display
        ignore_value: Value used to mark ignored/void regions (default: 250)
        **kwargs: Additional arguments (currently unused)
        
    Returns:
        Plotly figure for instance segmentation visualization
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # CRITICAL: Input validation with fail-fast assertions
    assert isinstance(instance_surrogate, torch.Tensor), f"Expected torch.Tensor, got {type(instance_surrogate)}"
    assert instance_surrogate.ndim in [3, 4], f"Expected 3D [2,H,W] or 4D [N,2,H,W] tensor, got shape {instance_surrogate.shape}"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    assert isinstance(ignore_value, int), f"Expected int ignore_value, got {type(ignore_value)}"
    assert instance_surrogate.numel() > 0, f"Instance surrogate tensor cannot be empty"
    
    # Handle batched input - extract single sample for visualization
    if instance_surrogate.ndim == 4:
        assert instance_surrogate.shape[0] == 1, f"Expected batch size 1 for visualization, got {instance_surrogate.shape[0]}"
        instance_surrogate = instance_surrogate[0]  # [N, 2, H, W] -> [2, H, W]
    
    # Validate unbatched tensor shape
    assert instance_surrogate.shape[0] == 2, f"Expected 2 channels [2, H, W], got {instance_surrogate.shape[0]} channels"
    
    # Extract Y and X offset channels
    y_offset = instance_surrogate[0]  # [H, W]
    x_offset = instance_surrogate[1]  # [H, W]
    
    # Convert coordinate representation to pseudo-instance mask for visualization
    instance_viz = _convert_surrogate_to_instance_mask(
        y_offset=y_offset,
        x_offset=x_offset,
        ignore_index=ignore_value
    )
    
    # Create the segmentation display with the converted format
    return create_segmentation_display(
        segmentation=instance_viz,
        title=title
    )


def get_instance_surrogate_display_stats(
    instance_surrogate: torch.Tensor,
    ignore_index: int = 250
) -> Dict[str, Any]:
    """Get statistics for instance surrogate representation.
    
    Args:
        instance_surrogate: Tensor of shape [2, H, W] or [N, 2, H, W] (batched) with coordinate offsets
        ignore_index: Value used to mark ignored/void regions (default: 250)
        
    Returns:
        Dictionary containing statistics about the instance surrogate data
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validation
    assert isinstance(instance_surrogate, torch.Tensor), f"Expected torch.Tensor, got {type(instance_surrogate)}"
    assert instance_surrogate.ndim in [3, 4], f"Expected 3D [2,H,W] or 4D [N,2,H,W] tensor, got shape {instance_surrogate.shape}"
    assert instance_surrogate.numel() > 0, f"Instance surrogate tensor cannot be empty"
    assert isinstance(ignore_index, int), f"Expected int ignore_index, got {type(ignore_index)}"
    
    # Handle batched input - extract single sample for analysis
    if instance_surrogate.ndim == 4:
        assert instance_surrogate.shape[0] == 1, f"Expected batch size 1 for analysis, got {instance_surrogate.shape[0]}"
        instance_surrogate = instance_surrogate[0]  # [N, 2, H, W] -> [2, H, W]
    
    # Validate unbatched tensor shape
    assert instance_surrogate.shape[0] == 2, f"Expected 2 channels [2, H, W], got {instance_surrogate.shape[0]} channels"
    
    # Extract Y and X offset channels
    y_offset = instance_surrogate[0]  # [H, W]
    x_offset = instance_surrogate[1]  # [H, W]
    
    # Create masks for analysis
    ignore_mask = (y_offset == ignore_index) & (x_offset == ignore_index)
    valid_mask = ~ignore_mask
    
    # Basic tensor statistics
    height, width = instance_surrogate.shape[1], instance_surrogate.shape[2]
    total_pixels = height * width
    valid_pixels = valid_mask.sum().item()
    ignore_pixels = ignore_mask.sum().item()
    
    stats = {
        "Shape": f"{instance_surrogate.shape}",
        "Height": height,
        "Width": width,
        "Total Pixels": total_pixels,
        "Valid Pixels": valid_pixels,
        "Ignore Pixels": ignore_pixels,
        "Valid Ratio": f"{valid_pixels / total_pixels:.3f}" if total_pixels > 0 else "0.000",
        "Data Type": str(instance_surrogate.dtype),
    }
    
    # Statistics for valid regions only
    if valid_pixels > 0:
        valid_y = y_offset[valid_mask]
        valid_x = x_offset[valid_mask]
        
        # Offset magnitude statistics
        magnitude = torch.sqrt(valid_y**2 + valid_x**2)
        
        stats.update({
            "Y Offset Range": f"[{valid_y.min().item():.3f}, {valid_y.max().item():.3f}]",
            "X Offset Range": f"[{valid_x.min().item():.3f}, {valid_x.max().item():.3f}]",
            "Y Offset Mean": f"{valid_y.mean().item():.3f}",
            "X Offset Mean": f"{valid_x.mean().item():.3f}",
            "Y Offset Std": f"{valid_y.std().item():.3f}",
            "X Offset Std": f"{valid_x.std().item():.3f}",
            "Magnitude Mean": f"{magnitude.mean().item():.3f}",
            "Magnitude Std": f"{magnitude.std().item():.3f}",
            "Max Magnitude": f"{magnitude.max().item():.3f}",
        })
    else:
        stats.update({
            "Y Offset Range": "N/A (no valid pixels)",
            "X Offset Range": "N/A (no valid pixels)",
            "Y Offset Mean": "N/A",
            "X Offset Mean": "N/A",
            "Y Offset Std": "N/A",
            "X Offset Std": "N/A",
            "Magnitude Mean": "N/A",
            "Magnitude Std": "N/A",
            "Max Magnitude": "N/A",
        })
    
    return stats


def _convert_surrogate_to_instance_mask(
    y_offset: torch.Tensor,
    x_offset: torch.Tensor,
    ignore_index: int
) -> torch.Tensor:
    """Convert coordinate surrogate representation to pseudo-instance mask.
    
    This function converts the 2-channel coordinate representation to a single-channel
    pseudo-instance mask for visualization purposes. The conversion creates discrete
    instance-like regions based on the offset magnitude patterns.
    
    Args:
        y_offset: Y-coordinate offsets of shape [H, W]
        x_offset: X-coordinate offsets of shape [H, W]
        ignore_index: Value used to mark ignored/void regions
        
    Returns:
        Single-channel pseudo-instance mask of shape [H, W] with int64 dtype
    """
    # Compute magnitude of offset vectors
    magnitude = torch.sqrt(y_offset**2 + x_offset**2)  # [H, W]
    
    # Handle ignore regions (where both offsets are equal to ignore_index)
    ignore_mask = (y_offset == ignore_index) & (x_offset == ignore_index)
    
    # Initialize output tensor
    instance_viz = torch.zeros_like(magnitude, dtype=torch.int64)
    
    # Only process non-ignore regions
    valid_mask = ~ignore_mask
    if valid_mask.any():
        valid_magnitude = magnitude[valid_mask]
        
        # Create discrete instance IDs based on magnitude quantiles
        # This gives a pseudo-instance visualization that reveals offset patterns
        if valid_magnitude.numel() > 0:
            # Use percentiles to create discrete levels (20 levels for good granularity)
            try:
                percentiles = torch.quantile(valid_magnitude, torch.linspace(0, 1, 20, device=valid_magnitude.device))
            except RuntimeError:
                # Fallback if quantile fails (e.g., all values are the same)
                percentiles = torch.tensor([valid_magnitude.min(), valid_magnitude.max()], device=valid_magnitude.device)
            
            # Map each magnitude to its percentile bin
            for i in range(len(percentiles) - 1):
                if i == len(percentiles) - 2:
                    # Include the maximum value in the last bin
                    mask = (magnitude >= percentiles[i]) & valid_mask
                else:
                    mask = (magnitude >= percentiles[i]) & (magnitude < percentiles[i + 1]) & valid_mask
                instance_viz[mask] = i + 1
    
    # Set ignore regions to ignore_index
    instance_viz[ignore_mask] = ignore_index
    
    return instance_viz