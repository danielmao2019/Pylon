"""Utility functions for point cloud visualization."""
from typing import Dict, Optional, Union, Any, Tuple
import time
import numpy as np
import torch
from dash import html
import plotly.graph_objects as go
from data.viewer.utils.segmentation import get_color
from data.viewer.utils.camera_lod import get_lod_manager


def _apply_lod_to_point_cloud(
    points: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]],
    labels: Optional[Union[torch.Tensor, np.ndarray]],
    camera_state: Dict[str, Any],
    point_cloud_id: str,
    title: str,
    lod_level: Optional[int] = None
) -> Tuple[Union[torch.Tensor, np.ndarray], Optional[Union[torch.Tensor, np.ndarray]], Optional[Union[torch.Tensor, np.ndarray]], str, float]:
    """Apply LOD processing to point cloud data.
    
    Args:
        points: Point cloud positions
        colors: Optional color data
        labels: Optional label data  
        camera_state: Camera viewing state
        point_cloud_id: Unique identifier for caching
        title: Title for debug logging
        lod_level: Force specific LOD level, None for automatic
        
    Returns:
        Tuple of (processed_points, processed_colors, processed_labels, updated_title, lod_time_ms)
    """
    original_point_count = len(points) if isinstance(points, (np.ndarray, list)) else points.shape[0]
    
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
            
    # Get the global LOD manager instance for proper caching
    lod_manager = get_lod_manager()
    
    lod_start = time.time()
    
    # Force specific LOD level if provided (backward compatibility)
    if lod_level is not None:
        # Convert old LOD levels to approximate target points for compatibility
        level_to_points = {0: original_point_count, 1: 50000, 2: 25000, 3: 10000}
        target_points = min(level_to_points.get(lod_level, original_point_count), original_point_count)
        print(f"[LOD DEBUG] {title} - FORCED LOD level {lod_level} → target: {target_points:,} points")
    else:
        # Calculate target points based on viewing conditions
        target_points = lod_manager.calculate_target_points(
            pc_dict, camera_state, point_cloud_id
        )
        
    # Validate target points
    target_points = max(2000, min(target_points, original_point_count))
    reduction_pct = (1 - target_points/original_point_count) * 100
    print(f"[LOD DEBUG] {title} - Original: {original_point_count:,}, Target: {target_points:,} points ({reduction_pct:.1f}% reduction)")
    
    # Apply downsampling only if meaningful reduction
    if target_points < original_point_count * 0.95:  # Only if reducing by >5%
        downsampled_pc = lod_manager.get_downsampled_point_cloud(
            pc_dict, target_points, point_cloud_id
        )
    else:
        # Skip LOD for minimal reduction to avoid overhead
        downsampled_pc = pc_dict
    lod_time = time.time() - lod_start
    
    # Convert back to numpy
    processed_points = point_cloud_to_numpy(downsampled_pc['pos'])
    processed_colors = colors  # Preserve original if not in downsampled
    processed_labels = labels  # Preserve original if not in downsampled
    
    if 'rgb' in downsampled_pc:
        processed_colors = point_cloud_to_numpy(downsampled_pc['rgb'])
    if 'labels' in downsampled_pc:
        processed_labels = point_cloud_to_numpy(downsampled_pc['labels'])
        
    # Update title with LOD info and performance analysis
    current_point_count = len(processed_points)
    actual_reduction = (original_point_count - current_point_count) / original_point_count
    
    # Performance analysis
    lod_overhead_ms = lod_time * 1000
    estimated_rendering_speedup = max(1.0, original_point_count / current_point_count)
    
    print(f"[LOD DEBUG] {title} - Final: {current_point_count:,} points ({actual_reduction*100:.1f}% reduction)")
    print(f"[LOD PERF] {title} - LOD overhead: {lod_overhead_ms:.1f}ms, Est. speedup: {estimated_rendering_speedup:.1f}x")
    
    updated_title = title
    if current_point_count != original_point_count:
        updated_title += f" (LOD: {current_point_count:,}/{original_point_count:,})"
        
    return processed_points, processed_colors, processed_labels, updated_title, lod_overhead_ms




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
    # Store original point count for logging
    original_point_count = len(points) if isinstance(points, (np.ndarray, list)) else points.shape[0]
    
    start_time = time.time()
    print(f"[LOD DEBUG] {title} - Original points: {original_point_count:,}, LOD enabled: {lod_enabled}, Camera state: {camera_state is not None}, PC ID: {point_cloud_id}")
    
    # Apply LOD if enabled and required conditions are met
    if lod_enabled and camera_state is not None and point_cloud_id is not None:
        points, colors, labels, title, _ = _apply_lod_to_point_cloud(
            points, colors, labels, camera_state, point_cloud_id, title, lod_level
        )
    else:
        print(f"[LOD DEBUG] {title} - LOD NOT APPLIED - LOD enabled: {lod_enabled}, Camera state: {camera_state is not None}, PC ID: {point_cloud_id}")
    
    # Convert input data to numpy arrays and validate
    points = point_cloud_to_numpy(points)
    
    # Validate point cloud is not empty
    if len(points) == 0:
        print(f"[LOD WARNING] {title} - Empty point cloud provided")
        # Create a minimal point cloud for display
        points = np.array([[0, 0, 0]], dtype=np.float32)
    
    # Process colors and labels
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

    total_time = time.time() - start_time
    print(f"[LOD DEBUG] {title} - TOTAL create_point_cloud_figure time: {total_time:.3f}s")
    
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
