"""Utility functions for point cloud visualization."""
from typing import Dict, Optional, Union, Any, Tuple
import numpy as np
import torch
from dash import html
import plotly.graph_objects as go
from data.viewer.utils.segmentation import get_color
from data.viewer.utils.continuous_lod import ContinuousLOD
from data.viewer.utils.discrete_lod import DiscreteLOD


def point_cloud_to_numpy(points: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a PyTorch tensor to a numpy array."""
    if isinstance(points, torch.Tensor):
        return points.cpu().numpy()
    return points


def apply_lod_to_point_cloud(
    points: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_type: Optional[str] = None,
    lod_config: Optional[Dict[str, Any]] = None,
    point_cloud_id: Optional[str] = None,
) -> Tuple[Union[torch.Tensor, np.ndarray], Optional[Union[torch.Tensor, np.ndarray]], Optional[Union[torch.Tensor, np.ndarray]]]:
    """Apply Level of Detail processing to point cloud data.
    
    Args:
        points: Point cloud positions
        colors: Optional color data
        labels: Optional label data
        camera_state: Camera viewing state
        lod_type: Type of LOD ("continuous", "discrete", or None)
        lod_config: Optional LOD configuration parameters
        point_cloud_id: Unique identifier for discrete LOD caching
        
    Returns:
        Tuple of (processed_points, processed_colors, processed_labels)
    """
    # If no LOD requested, return originals
    if lod_type is None or camera_state is None:
        return points, colors, labels
        
    # Convert to numpy for consistency
    points = point_cloud_to_numpy(points)
    
    # Prepare point cloud dictionary
    pc_dict = {'pos': torch.from_numpy(points).float()}
    if colors is not None:
        pc_dict['rgb'] = torch.from_numpy(point_cloud_to_numpy(colors))
    if labels is not None:
        pc_dict['labels'] = torch.from_numpy(point_cloud_to_numpy(labels))
    
    # Apply LOD based on type
    if lod_type == "continuous":
        # Continuous LOD with distance-based sampling
        lod = ContinuousLOD(**(lod_config or {}))
        downsampled = lod.subsample(pc_dict, camera_state)
    elif lod_type == "discrete" and point_cloud_id is not None:
        # Discrete LOD with pre-computed levels
        lod = DiscreteLOD(**(lod_config or {}))
        if not lod.has_levels(point_cloud_id):
            lod.precompute_levels(pc_dict, point_cloud_id)
        downsampled = lod.select_level(point_cloud_id, camera_state)
    else:
        downsampled = pc_dict
        
    # Extract downsampled data
    processed_points = point_cloud_to_numpy(downsampled['pos'])
    processed_colors = (
        point_cloud_to_numpy(downsampled['rgb']) if 'rgb' in downsampled 
        else colors
    )
    processed_labels = (
        point_cloud_to_numpy(downsampled['labels']) if 'labels' in downsampled 
        else labels
    )
    
    return processed_points, processed_colors, processed_labels


def create_point_cloud_figure(
    points: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Point Cloud",
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_type: Optional[str] = "continuous",
    lod_config: Optional[Dict[str, Any]] = None,
    point_cloud_id: Optional[str] = None,
) -> go.Figure:
    """Create a 3D point cloud visualization figure with optional LOD.

    Args:
        points: Numpy array or tensor of shape (N, 3) containing XYZ coordinates
        colors: Optional numpy array of shape (N, 3) containing RGB color values
        labels: Optional numpy array of shape (N,) containing labels
        title: Title for the figure
        point_size: Size of the points
        point_opacity: Opacity of the points
        camera_state: Optional dictionary containing camera position state
        lod_type: Type of LOD ("continuous", "discrete", or None for no LOD)
        lod_config: Optional LOD configuration parameters
        point_cloud_id: Unique identifier for discrete LOD caching

    Returns:
        Plotly Figure object with potentially downsampled point cloud
    """
    # Convert to numpy for consistency
    points = point_cloud_to_numpy(points)
    original_count = len(points)
    
    # Apply LOD using helper function
    points, colors, labels = apply_lod_to_point_cloud(
        points=points,
        colors=colors,
        labels=labels,
        camera_state=camera_state,
        lod_type=lod_type,
        lod_config=lod_config,
        point_cloud_id=point_cloud_id
    )
    
    # Update title with LOD info
    if len(points) < original_count:
        lod_suffix = f" ({lod_type.title()} LOD: {len(points):,}/{original_count:,})"
        title = f"{title}{lod_suffix}"
    
    # Handle edge case of empty point clouds
    if len(points) == 0:
        points = np.array([[0, 0, 0]], dtype=np.float32)
    
    # Process colors from labels if needed
    if colors is not None:
        colors = point_cloud_to_numpy(colors)
        assert colors.shape == points.shape, f"{colors.shape=}, {points.shape=}"
    elif labels is not None:
        labels = point_cloud_to_numpy(labels)
        assert labels.shape == points.shape[:-1], f"{labels.shape=}, {points.shape=}"
        # Convert labels to colors
        unique_labels = np.unique(labels)
        unique_colors = [get_color(label) for label in unique_labels]
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        for label, color in zip(unique_labels, unique_colors):
            mask = labels == label
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            colors[mask, :] = np.array([r, g, b], dtype=np.uint8)
    
    # Create Plotly scatter plot
    scatter3d_kwargs = dict(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=point_size, opacity=point_opacity),
        hoverinfo='skip',  # Disable hover for performance
    )
    
    if colors is not None:
        scatter3d_kwargs['marker']['color'] = colors
    else:
        scatter3d_kwargs['marker']['color'] = 'steelblue'
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(**scatter3d_kwargs))
    
    # Calculate bounding box
    x_range = [points[:, 0].min(), points[:, 0].max()]
    y_range = [points[:, 1].min(), points[:, 1].max()]
    z_range = [points[:, 2].min(), points[:, 2].max()]
    
    # Set layout
    camera = camera_state if camera_state else {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
    }
    
    fig.update_layout(
        title=title,
        uirevision='camera',  # This ensures camera views stay in sync
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=camera,
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range)
        ),
        margin=dict(l=0, r=40, b=0, t=40),
        height=500,
    )
    
    return fig


def get_point_cloud_stats(
    points: torch.Tensor,
    change_map: Optional[torch.Tensor] = None,
    class_names: Optional[Dict[int, str]] = None
) -> html.Ul:
    """Get statistical information about a point cloud.

    Args:
        points: Point cloud tensor of shape (N, 3+)
        change_map: Optional tensor with change classes for each point
        class_names: Optional dictionary mapping class IDs to class names

    Returns:
        List of html components with point cloud statistics
    """
    # Basic stats
    points_np = points.detach().cpu().numpy()
    stats_items = [
        html.Li(f"Total Points: {len(points_np)}"),
        html.Li(f"Dimensions: {points_np.shape[1]}"),
        html.Li(f"X Range: [{points_np[:, 0].min():.2f}, {points_np[:, 0].max():.2f}]"),
        html.Li(f"Y Range: [{points_np[:, 1].min():.2f}, {points_np[:, 1].max():.2f}]"),
        html.Li(f"Z Range: [{points_np[:, 2].min():.2f}, {points_np[:, 2].max():.2f}]"),
        html.Li(f"Center: [{points_np[:, 0].mean():.2f}, {points_np[:, 1].mean():.2f}, {points_np[:, 2].mean():.2f}]")
    ]

    # Add class distribution if change_map is provided
    if change_map is not None:
        unique_classes, class_counts = torch.unique(change_map, return_counts=True)
        unique_classes = unique_classes.cpu().numpy()
        class_counts = class_counts.cpu().numpy()
        total_points = change_map.numel()

        stats_items.append(html.Li("Class Distribution:"))
        class_list_items = []

        for cls, count in zip(unique_classes, class_counts):
            percentage = (count / total_points) * 100
            cls_key = cls.item() if hasattr(cls, 'item') else cls
            class_name = class_names[cls_key] if class_names and cls_key in class_names else f"Class {cls_key}"
            class_list_items.append(
                html.Li(f"{class_name}: {count} points ({percentage:.2f}%)",
                       style={'marginLeft': '20px'})
            )

        stats_items.append(html.Ul(class_list_items))

    return html.Ul(stats_items)