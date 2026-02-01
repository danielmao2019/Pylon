"""Edge display utilities for edge detection visualization."""

from typing import Any, Dict

import plotly.express as px
import plotly.graph_objects as go
import torch


def create_edge_display(
    edges: torch.Tensor, title: str, colorscale: str = 'Gray', **kwargs: Any
) -> go.Figure:
    """Create edge detection display with proper visualization.

    Args:
        edges: Edge tensor of shape [H, W] or [N, H, W] (batched) with edge values
        title: Title for the edge display
        colorscale: Plotly colorscale for edge visualization
        **kwargs: Additional arguments

    Returns:
        Plotly figure for edge visualization

    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validations
    assert isinstance(edges, torch.Tensor), f"{type(edges)=}"
    assert edges.ndim in [2, 3], f"{edges.shape=}"
    assert edges.numel() > 0, f"{edges.shape=}"
    assert isinstance(title, str), f"{type(title)=}"
    assert isinstance(colorscale, str), f"{type(colorscale)=}"
    assert edges.ndim != 3 or edges.shape[0] == 1, f"{edges.shape=}"

    # Input normalizations
    if edges.ndim == 3:
        edges = edges[0]  # [N, H, W] -> [H, W]

    # Convert to numpy for visualization
    edges_np = edges.detach().cpu().numpy()

    # Create edge visualization
    fig = px.imshow(
        img=edges_np,
        color_continuous_scale=colorscale,
        title=title,
        labels={'color': 'Edge Strength'},
    )

    # Update layout for better visualization
    fig.update_layout(title_x=0.5, coloraxis_colorbar=dict(title="Edge Strength"))

    return fig


def get_edge_display_stats(edges: torch.Tensor) -> Dict[str, Any]:
    """Get edge detection statistics for display.

    Args:
        edges: Edge tensor of shape [H, W] or [N, H, W] (batched)

    Returns:
        Dictionary containing edge statistics

    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validations
    assert isinstance(edges, torch.Tensor), f"{type(edges)=}"
    assert edges.ndim in [2, 3], f"{edges.shape=}"
    assert edges.numel() > 0, f"{edges.shape=}"
    assert edges.ndim != 3 or edges.shape[0] == 1, f"{edges.shape=}"

    # Input normalizations
    if edges.ndim == 3:
        edges = edges[0]  # [N, H, W] -> [H, W]

    # Calculate statistics
    valid_mask = torch.isfinite(edges)
    valid_edges = edges[valid_mask]

    if len(valid_edges) == 0:
        return {
            'shape': list(edges.shape),
            'dtype': str(edges.dtype),
            'valid_pixels': 0,
            'total_pixels': edges.numel(),
            'min_edge': 'N/A',
            'max_edge': 'N/A',
            'mean_edge': 'N/A',
            'std_edge': 'N/A',
            'edge_percentage': 'N/A',
        }

    # Calculate edge statistics
    # For binary edges, count non-zero as edge pixels
    edge_threshold = (
        0.5 if edges.dtype in [torch.float32, torch.float64, torch.float16] else 0
    )
    edge_pixels = (valid_edges > edge_threshold).sum()
    edge_percentage = (float(edge_pixels) / len(valid_edges)) * 100

    # Convert to float for statistical calculations if needed (handles integer dtypes)
    if valid_edges.dtype in [
        torch.int32,
        torch.int64,
        torch.int16,
        torch.int8,
        torch.uint8,
    ]:
        valid_edges_float = valid_edges.float()
    else:
        valid_edges_float = valid_edges

    return {
        'shape': list(edges.shape),
        'dtype': str(edges.dtype),
        'valid_pixels': len(valid_edges),
        'total_pixels': edges.numel(),
        'min_edge': float(valid_edges.min()),
        'max_edge': float(valid_edges.max()),
        'mean_edge': float(valid_edges_float.mean()),
        'std_edge': float(valid_edges_float.std()),
        'edge_percentage': edge_percentage,
    }
