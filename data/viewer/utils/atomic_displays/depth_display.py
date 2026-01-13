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
    ignore_value: Optional[float] = None,
    **kwargs: Any
) -> go.Figure:
    """Create depth map display with proper visualization.

    Args:
        depth: Depth tensor of shape [H, W] or [N, H, W] (batched) with depth values
        title: Title for the depth display
        colorscale: Plotly colorscale for depth visualization
        ignore_value: Optional value to treat as invalid/transparent
        **kwargs: Additional arguments

    Returns:
        Plotly figure for depth visualization

    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # CRITICAL: Input validation with fail-fast assertions
    assert isinstance(depth, torch.Tensor), f"Expected torch.Tensor, got {type(depth)}"
    assert depth.ndim in [2, 3], f"Expected 2D [H,W] or 3D [N,H,W] tensor, got shape {depth.shape}"
    assert depth.numel() > 0, f"Depth tensor cannot be empty"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    assert isinstance(colorscale, str), f"Expected str colorscale, got {type(colorscale)}"

    # Handle batched input - extract single sample for visualization
    if depth.ndim == 3:
        assert depth.shape[0] == 1, f"Expected batch size 1 for visualization, got {depth.shape[0]}"
        depth = depth[0]  # [N, H, W] -> [H, W]

    # Convert to numpy for visualization
    depth_np = depth.detach().cpu().numpy()

    # Handle ignore values with custom coloring
    if ignore_value is not None:
        # Create mask for ignore values
        ignore_mask = np.abs(depth_np - ignore_value) < 1e-5
        valid_mask = ~ignore_mask

        if np.any(ignore_mask):
            # Create custom visualization with separate colors for ignore values
            fig = go.Figure()

            # Add valid depth values with main colorscale
            if np.any(valid_mask):
                valid_depth = depth_np.copy()
                valid_depth[ignore_mask] = np.nan  # Hide ignore values for this trace

                fig.add_trace(go.Heatmap(
                    z=valid_depth,
                    colorscale=colorscale,
                    name='Valid Depth',
                    showscale=True,
                    colorbar=dict(title="Depth (m)", x=1.0)
                ))

            # Add ignore values with subtle gray color
            if np.any(ignore_mask):
                ignore_depth = np.full_like(depth_np, np.nan)
                ignore_depth[ignore_mask] = 0  # Use 0 as placeholder for consistent coloring

                fig.add_trace(go.Heatmap(
                    z=ignore_depth,
                    colorscale=[[0, 'lightgray'], [1, 'lightgray']],  # Subtle gray for ignore values
                    name='Ignore Values',
                    showscale=False,  # Don't show colorbar for ignore values
                    opacity=0.7  # Slightly transparent
                ))

            # Update layout to match image display behavior
            fig.update_layout(
                title=title,
                title_x=0.5,
                margin=dict(l=20, r=20, t=40, b=20),
                height=400,
                xaxis=dict(showticklabels=True, scaleanchor="y", scaleratio=1),  # Lock aspect ratio
                yaxis=dict(showticklabels=True, autorange='reversed')  # Enable tick labels and flip y-axis
            )
        else:
            # No ignore values present, use standard visualization
            fig = px.imshow(
                depth_np,
                color_continuous_scale=colorscale,
                title=title,
                labels={'color': 'Depth'}
            )
    else:
        # No ignore value specified, use standard visualization
        fig = px.imshow(
            depth_np,
            color_continuous_scale=colorscale,
            title=title,
            labels={'color': 'Depth'}
        )

    # Ensure layout is properly set for standard visualizations
    if ignore_value is None or not np.any(np.abs(depth_np - ignore_value) < 1e-5):
        fig.update_layout(
            title_x=0.5,
            margin=dict(l=20, r=20, t=40, b=20),
            height=400,
            coloraxis_colorbar=dict(title="Depth"),
            coloraxis_showscale=True,
            showlegend=False,
            xaxis=dict(scaleanchor="y", scaleratio=1),  # Lock aspect ratio
            yaxis=dict(autorange='reversed')  # Standard image convention
        )

    return fig


def get_depth_display_stats(
    depth: torch.Tensor,
    ignore_value: Optional[float] = None
) -> Dict[str, Any]:
    """Get depth statistics for display.

    Args:
        depth: Depth tensor of shape [H, W] or [N, H, W] (batched)
        ignore_value: Optional value to ignore when calculating statistics

    Returns:
        Dictionary containing depth statistics

    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validation
    assert isinstance(depth, torch.Tensor), f"Expected torch.Tensor, got {type(depth)}"
    assert depth.ndim in [2, 3], f"Expected 2D [H,W] or 3D [N,H,W] tensor, got shape {depth.shape}"
    assert depth.numel() > 0, f"Depth tensor cannot be empty"

    # Handle batched input - extract single sample for analysis
    if depth.ndim == 3:
        assert depth.shape[0] == 1, f"Expected batch size 1 for analysis, got {depth.shape[0]}"
        depth = depth[0]  # [N, H, W] -> [H, W]

    # Calculate statistics
    valid_mask = torch.isfinite(depth) & (depth > 0)

    # Filter out ignore_value if specified
    if ignore_value is not None:
        if ignore_value < 0:
            # For negative ignore values, simple comparison is sufficient
            valid_mask = valid_mask & (depth >= 0)
        else:
            # For positive ignore values, use epsilon-based comparison to handle floating point precision
            valid_mask = valid_mask & (torch.abs(depth - ignore_value) > 1e-3)

    valid_depth = depth[valid_mask]

    if len(valid_depth) == 0:
        return {
            'shape': list(depth.shape),
            'dtype': str(depth.dtype),
            'valid_pixels': 0,
            'total_pixels': depth.numel(),
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
