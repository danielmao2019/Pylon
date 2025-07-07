"""Utility functions for point cloud visualization."""
from typing import Dict, Optional, Union, Any, Tuple
import time
import numpy as np
import torch
from dash import html
import plotly.graph_objects as go
from data.viewer.utils.segmentation import get_color
from data.viewer.utils.camera_lod import get_lod_manager


def _convert_to_tensor(data: Optional[Union[torch.Tensor, np.ndarray]]) -> Optional[torch.Tensor]:
    """Convert numpy array or tensor to tensor format."""
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    return data


def _get_point_count(points: Union[torch.Tensor, np.ndarray]) -> int:
    """Get point count from tensor or array."""
    return len(points) if isinstance(points, (np.ndarray, list)) else points.shape[0]


def _prepare_point_cloud_dict(
    points: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]],
    labels: Optional[Union[torch.Tensor, np.ndarray]]
) -> Dict[str, torch.Tensor]:
    """Prepare point cloud dictionary for LOD processing."""
    pc_dict = {'pos': _convert_to_tensor(points).float()}
    
    colors_tensor = _convert_to_tensor(colors)
    if colors_tensor is not None:
        pc_dict['rgb'] = colors_tensor
        
    labels_tensor = _convert_to_tensor(labels)
    if labels_tensor is not None:
        pc_dict['labels'] = labels_tensor
        
    return pc_dict


def _calculate_target_points(
    pc_dict: Dict[str, torch.Tensor],
    camera_state: Dict[str, Any],
    point_cloud_id: str,
    lod_level: Optional[int],
    original_count: int
) -> int:
    """Calculate target points for LOD processing."""
    lod_manager = get_lod_manager()
    
    if lod_level is not None:
        # Convert old LOD levels to approximate target points for compatibility
        level_to_points = {0: original_count, 1: 50000, 2: 25000, 3: 10000}
        return min(level_to_points.get(lod_level, original_count), original_count)
    
    # Calculate target points based on viewing conditions
    target = lod_manager.calculate_target_points(pc_dict, camera_state, point_cloud_id)
    return max(2000, min(target, original_count))


def _apply_downsampling(
    pc_dict: Dict[str, torch.Tensor],
    target_points: int,
    point_cloud_id: str
) -> Dict[str, torch.Tensor]:
    """Apply downsampling if meaningful reduction is possible."""
    original_count = pc_dict['pos'].shape[0]
    
    # Only apply LOD if meaningful reduction (>5%)
    if target_points < original_count * 0.95:
        lod_manager = get_lod_manager()
        return lod_manager.get_downsampled_point_cloud(pc_dict, target_points, point_cloud_id)
    
    return pc_dict


def _extract_processed_data(
    downsampled_pc: Dict[str, torch.Tensor],
    original_colors: Optional[Union[torch.Tensor, np.ndarray]],
    original_labels: Optional[Union[torch.Tensor, np.ndarray]]
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract processed data from downsampled point cloud."""
    processed_points = point_cloud_to_numpy(downsampled_pc['pos'])
    
    # Use downsampled colors/labels if available, otherwise preserve originals
    processed_colors = original_colors
    if 'rgb' in downsampled_pc:
        processed_colors = point_cloud_to_numpy(downsampled_pc['rgb'])
        
    processed_labels = original_labels
    if 'labels' in downsampled_pc:
        processed_labels = point_cloud_to_numpy(downsampled_pc['labels'])
        
    return processed_points, processed_colors, processed_labels


def _create_updated_title(title: str, original_count: int, current_count: int) -> str:
    """Create updated title with LOD information."""
    if current_count != original_count:
        return f"{title} (LOD: {current_count:,}/{original_count:,})"
    return title


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
        title: Title for logging
        lod_level: Force specific LOD level, None for automatic
        
    Returns:
        Tuple of (processed_points, processed_colors, processed_labels, updated_title, lod_time_ms)
    """
    original_count = _get_point_count(points)
    pc_dict = _prepare_point_cloud_dict(points, colors, labels)
    
    lod_start = time.time()
    target_points = _calculate_target_points(pc_dict, camera_state, point_cloud_id, lod_level, original_count)
    downsampled_pc = _apply_downsampling(pc_dict, target_points, point_cloud_id)
    lod_time = time.time() - lod_start
    
    processed_data = _extract_processed_data(downsampled_pc, colors, labels)
    updated_title = _create_updated_title(title, original_count, len(processed_data[0]))
    
    return *processed_data, updated_title, lod_time * 1000




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
    start_time = time.time()
    
    # Apply LOD if enabled and required conditions are met
    if lod_enabled and camera_state is not None and point_cloud_id is not None:
        points, colors, labels, title, _ = _apply_lod_to_point_cloud(
            points, colors, labels, camera_state, point_cloud_id, title, lod_level
        )
    # LOD conditions not met - use original data
    
    # Convert input data to numpy arrays and validate
    points = point_cloud_to_numpy(points)
    
    # Handle empty point clouds
    if len(points) == 0:
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

    # Performance monitoring available if needed
    _ = time.time() - start_time
    
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
