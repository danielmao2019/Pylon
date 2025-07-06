"""Utility functions for point cloud visualization."""
from typing import Dict, Optional, Union, Any
import numpy as np
import torch
from dash import html
import plotly.graph_objects as go
from data.viewer.utils.segmentation import get_color
from data.viewer.utils.camera_lod import get_lod_manager, calculate_point_cloud_bounds


def point_cloud_to_numpy(points: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a PyTorch tensor to a displayable point cloud."""
    if isinstance(points, torch.Tensor):
        return points.cpu().numpy()
    return points


def create_point_cloud_figure(
    points: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Point Cloud",
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_enabled: bool = True,
    lod_level: Optional[int] = None,
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
        lod_enabled: Whether to enable Level of Detail optimization
        lod_level: Force specific LOD level (0-3), None for auto-calculation
        point_cloud_id: Unique identifier for LOD caching

    Returns:
        Plotly Figure object with potentially downsampled point cloud
    """
    # Store original point count for LOD info
    original_point_count = len(points) if isinstance(points, (np.ndarray, list)) else points.shape[0]
    
    # Apply LOD if enabled
    if lod_enabled and camera_state is not None and point_cloud_id is not None:
        # Convert to tensor format for LOD processing
        if isinstance(points, np.ndarray):
            points_tensor = torch.from_numpy(points).float()
        else:
            points_tensor = points.float()
            
        # Create point cloud dictionary for LOD manager
        pc_dict = {'pos': points_tensor}
        if colors is not None:
            if isinstance(colors, np.ndarray):
                pc_dict['rgb'] = torch.from_numpy(colors)
            else:
                pc_dict['rgb'] = colors
        if labels is not None:
            if isinstance(labels, np.ndarray):
                pc_dict['labels'] = torch.from_numpy(labels)
            else:
                pc_dict['labels'] = labels
                
        # Calculate camera distance and LOD level
        lod_manager = get_lod_manager()
        center, bounds = calculate_point_cloud_bounds(points_tensor)
        camera_distance = lod_manager.calculate_camera_distance(camera_state, center, bounds)
        auto_lod_level = lod_manager.get_lod_level(camera_distance, point_cloud_id, lod_level)
        
        # Apply LOD downsampling
        downsampled_pc = lod_manager.get_downsampled_point_cloud(pc_dict, auto_lod_level, point_cloud_id)
        
        # Convert back to numpy
        points = point_cloud_to_numpy(downsampled_pc['pos'])
        if 'rgb' in downsampled_pc:
            colors = point_cloud_to_numpy(downsampled_pc['rgb'])
        if 'labels' in downsampled_pc:
            labels = point_cloud_to_numpy(downsampled_pc['labels'])
            
        # Update title with LOD info
        current_point_count = len(points)
        if current_point_count != original_point_count:
            title += f" (LOD {auto_lod_level}: {current_point_count:,}/{original_point_count:,} points)"
    
    # Convert input data to numpy arrays
    points = point_cloud_to_numpy(points)
    if colors is not None:
        colors = point_cloud_to_numpy(colors)
        assert colors.shape == points.shape, f"{colors.shape=}, {points.shape=}"
    elif labels is not None:
        labels = point_cloud_to_numpy(labels)
        assert labels.shape == points.shape[:-1], f"{labels.shape=}, {points.shape=}"
        unique_labels = np.unique(labels)
        unique_colors = [get_color(label) for label in unique_labels]
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        for label, color in zip(unique_labels, unique_colors):
            mask = labels == label
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            colors[mask, :] = np.array([r, g, b], dtype=np.uint8)

    # Add point cloud
    scatter3d_kwargs = dict(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=point_size, opacity=point_opacity),
        hoverinfo='skip',  # Disable hover for performance - was causing massive memory overhead
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
