"""Point cloud display utilities for 3D point cloud visualization."""
from typing import Dict, Optional, Union, Any, Tuple, List
import torch
from dash import html
import plotly.graph_objects as go
from data.viewer.utils.point_cloud import (
    create_point_cloud_figure,
    get_point_cloud_stats,
    build_point_cloud_id
)


def create_point_cloud_display(
    points: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    title: str = "Point Cloud",
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_type: str = "continuous",
    lod_config: Optional[Dict[str, Any]] = None,
    density_percentage: Optional[int] = None,
    point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    **kwargs: Any
) -> go.Figure:
    """Create point cloud display with LOD optimization.
    
    Args:
        points: Point cloud positions tensor of shape [N, 3]
        colors: Optional color tensor of shape [N, 3] or [N, C]
        labels: Optional label tensor of shape [N]
        title: Title for the point cloud display
        point_size: Size of the points in visualization
        point_opacity: Opacity of the points in visualization
        camera_state: Optional camera position state for 3D view
        lod_type: Type of LOD ("continuous", "discrete", or "none")
        lod_config: Optional LOD configuration parameters
        density_percentage: Percentage of points to display when lod_type is "none" (1-100)
        point_cloud_id: Unique identifier for LOD caching
        axis_ranges: Optional fixed axis ranges for consistent scaling
        **kwargs: Additional arguments
        
    Returns:
        Plotly figure for point cloud visualization
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # CRITICAL: Input validation with fail-fast assertions
    assert isinstance(points, torch.Tensor), f"Expected torch.Tensor, got {type(points)}"
    assert points.ndim == 2, f"Expected 2D tensor [N,3], got shape {points.shape}"
    assert points.shape[1] == 3, f"Expected 3 coordinates, got {points.shape[1]}"
    assert points.numel() > 0, f"Point cloud cannot be empty"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    assert isinstance(point_size, (int, float)), f"Expected numeric point_size, got {type(point_size)}"
    assert isinstance(point_opacity, (int, float)), f"Expected numeric point_opacity, got {type(point_opacity)}"
    assert 0.0 <= point_opacity <= 1.0, f"point_opacity must be in [0,1], got {point_opacity}"
    
    if colors is not None:
        assert isinstance(colors, torch.Tensor), f"colors must be torch.Tensor, got {type(colors)}"
        assert colors.shape[0] == points.shape[0], f"colors length {colors.shape[0]} != points length {points.shape[0]}"
    
    if labels is not None:
        assert isinstance(labels, torch.Tensor), f"labels must be torch.Tensor, got {type(labels)}"
        assert labels.shape[0] == points.shape[0], f"labels length {labels.shape[0]} != points length {points.shape[0]}"
    
    if camera_state is not None:
        assert isinstance(camera_state, dict), f"camera_state must be dict, got {type(camera_state)}"
    
    if lod_config is not None:
        assert isinstance(lod_config, dict), f"lod_config must be dict, got {type(lod_config)}"
    
    if density_percentage is not None:
        assert isinstance(density_percentage, int), f"density_percentage must be int, got {type(density_percentage)}"
        assert 1 <= density_percentage <= 100, f"density_percentage must be 1-100, got {density_percentage}"
    
    if axis_ranges is not None:
        assert isinstance(axis_ranges, dict), f"axis_ranges must be dict, got {type(axis_ranges)}"
    
    # Use existing create_point_cloud_figure implementation
    return create_point_cloud_figure(
        points=points,
        colors=colors,
        labels=labels,
        title=title,
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state,
        lod_type=lod_type,
        lod_config=lod_config,
        density_percentage=density_percentage,
        point_cloud_id=point_cloud_id,
        axis_ranges=axis_ranges
    )


def get_point_cloud_display_stats(
    points: torch.Tensor,
    change_map: Optional[torch.Tensor] = None,
    class_names: Optional[Dict[int, str]] = None
) -> html.Ul:
    """Get point cloud statistics for display.
    
    Args:
        points: Point cloud tensor of shape [N, 3+]
        change_map: Optional tensor with change classes for each point
        class_names: Optional dictionary mapping class IDs to class names
        
    Returns:
        HTML list containing point cloud statistics
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validation
    assert isinstance(points, torch.Tensor), f"Expected torch.Tensor, got {type(points)}"
    assert points.ndim == 2, f"Expected 2D tensor [N,D], got shape {points.shape}"
    assert points.shape[1] >= 3, f"Expected at least 3 coordinates, got {points.shape[1]}"
    assert points.numel() > 0, f"Point cloud cannot be empty"
    
    if change_map is not None:
        assert isinstance(change_map, torch.Tensor), f"change_map must be torch.Tensor, got {type(change_map)}"
        assert change_map.shape[0] == points.shape[0], f"change_map length {change_map.shape[0]} != points length {points.shape[0]}"
    
    if class_names is not None:
        assert isinstance(class_names, dict), f"class_names must be dict, got {type(class_names)}"
    
    # Use existing get_point_cloud_stats implementation
    return get_point_cloud_stats(
        points=points,
        change_map=change_map,
        class_names=class_names
    )


def build_point_cloud_display_id(
    datapoint: Dict[str, Any], 
    component: str
) -> Tuple[str, int, str]:
    """Build structured point cloud ID for LOD caching.
    
    Args:
        datapoint: Contains meta_info with dataset context
        component: Point cloud component identifier
        
    Returns:
        Tuple of (dataset_name, datapoint_idx, component)
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    assert isinstance(datapoint, dict), f"datapoint must be dict, got {type(datapoint)}"
    assert isinstance(component, str), f"component must be str, got {type(component)}"
    
    # Use existing build_point_cloud_id implementation
    return build_point_cloud_id(datapoint=datapoint, component=component)