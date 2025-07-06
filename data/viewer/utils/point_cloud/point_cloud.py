"""Point cloud visualization utilities using Plotly."""
from typing import Dict, Optional, Union, Any
import numpy as np
import torch
from dash import html, dcc
import plotly.graph_objects as go
from data.viewer.utils.segmentation import get_color

# Global registry for tracking created graphs for camera sync
_created_graphs = []


def point_cloud_to_numpy(points: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a PyTorch tensor to a displayable point cloud."""
    if isinstance(points, torch.Tensor):
        return points.cpu().numpy()
    return points


def prepare_point_cloud_data(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Prepare point cloud data for rendering."""
    # Ensure points are float32
    points = points.astype(np.float32)

    # Handle colors
    if colors is not None:
        colors = colors.astype(np.float32)
        # Normalize colors to [0, 1] if they're in [0, 255] range
        if colors.max() > 1.0:
            colors = colors / 255.0
    elif labels is not None:
        # Convert labels to colors
        unique_labels = np.unique(labels)
        colors = np.zeros((len(points), 3), dtype=np.float32)

        for label in unique_labels:
            mask = labels == label
            color_hex = get_color(label)
            # Convert hex color to RGB
            r = int(color_hex[1:3], 16) / 255.0
            g = int(color_hex[3:5], 16) / 255.0
            b = int(color_hex[5:7], 16) / 255.0
            colors[mask] = [r, g, b]
    else:
        # Default blue color
        colors = np.full((len(points), 3), [0.27, 0.51, 0.71], dtype=np.float32)  # steelblue

    # Transform points to match working example coordinate range [-0.5, 0.5]
    # Working example generates points in this range directly
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = np.max(bbox_max - bbox_min)
    
    # Transform to [-0.5, 0.5] range like working example
    points_transformed = (points - bbox_center) / bbox_size

    return {
        'positions': points_transformed.flatten().tolist(),  # Transform to working range
        'colors': colors.flatten().tolist(),
        'point_count': len(points),
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'bbox_center': bbox_center.tolist(),
        'bbox_size': float(bbox_size)
    }




def create_point_cloud_figure(
    points: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Point Cloud",
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
) -> html.Div:
    """Create a point cloud visualization with Plotly and synchronized cameras.

    Args:
        points: Numpy array of shape (N, 3) containing XYZ coordinates
        colors: Optional numpy array of shape (N, 3) containing RGB color values
        labels: Optional numpy array of shape (N,) containing labels
        title: Title for the figure
        point_size: Size of the points
        point_opacity: Opacity of the points
        camera_state: Optional dictionary containing camera position state

    Returns:
        Plotly Graph component with synchronized camera
    """
    # Convert inputs to numpy
    points = point_cloud_to_numpy(points)
    if colors is not None:
        colors = point_cloud_to_numpy(colors)
    if labels is not None:
        labels = point_cloud_to_numpy(labels)

    # Prepare data for rendering
    point_cloud_data = prepare_point_cloud_data(points, colors, labels)

    # Use consistent graph indices based on title to avoid mixing
    graph_index_map = {
        "Source Point Cloud": 0,
        "Target Point Cloud": 1,
        "Union (Transformed Source + Target)": 2,
        "Symmetric Difference": 3,
        "Symmetric Difference (Empty)": 3  # Same index as regular symmetric difference
    }
    
    # Get graph index from title, default to auto-increment for unknown titles
    if title in graph_index_map:
        graph_index = graph_index_map[title]
    else:
        # For other titles, use auto-increment
        if not hasattr(create_point_cloud_figure, '_graph_counter'):
            create_point_cloud_figure._graph_counter = 4  # Start after known indices
        graph_index = create_point_cloud_figure._graph_counter
        create_point_cloud_figure._graph_counter += 1
    
    # Create synchronized Plotly visualization
    return _create_plotly_point_cloud(
        point_cloud_data, 
        title, 
        point_size, 
        point_opacity,
        graph_index=graph_index,
        camera_state=camera_state
    )


def _create_plotly_point_cloud(point_cloud_data: Dict[str, Any], title: str, point_size: float, point_opacity: float, graph_index: int = 0, camera_state: Optional[Dict[str, Any]] = None) -> dcc.Graph:
    """Create a point cloud using Plotly.
    
    Args:
        point_cloud_data: Prepared point cloud data from prepare_point_cloud_data()
        title: Title for the visualization
        point_size: Size of the points
        point_opacity: Opacity of the points
        graph_index: Index for camera synchronization
        camera_state: Optional camera state for synchronization
        
    Returns:
        Plotly Graph component with synchronized camera
    """
    
    # Extract data
    positions = np.array(point_cloud_data['positions']).reshape(-1, 3)
    colors = np.array(point_cloud_data['colors']).reshape(-1, 3)
    point_count = point_cloud_data['point_count']
    
    
    # Convert colors to RGB strings
    color_strings = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]
    
    # Create 3D scatter plot with WebGL acceleration
    fig = go.Figure(data=[go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1], 
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=max(2, point_size),
            color=color_strings,
            opacity=point_opacity,
        ),
        text=[f'Point {i}' for i in range(len(positions))],
        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
    )])
    
    # Configure camera - convert from legacy format to Plotly format if needed
    default_camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.5),
        center=dict(x=0, y=0, z=0), 
        up=dict(x=0, y=0, z=1)
    )
    
    # Convert camera_state from callback format to Plotly format
    if camera_state and isinstance(camera_state, dict):
        if 'eye' in camera_state:
            # Already in Plotly format
            camera_config = camera_state
        elif 'position' in camera_state and 'target' in camera_state:
            # Convert from legacy position/target format to Plotly eye/center format
            camera_config = dict(
                eye=dict(x=camera_state['position'][0], y=camera_state['position'][1], z=camera_state['position'][2]),
                center=dict(x=camera_state['target'][0], y=camera_state['target'][1], z=camera_state['target'][2]),
                up=dict(x=0, y=0, z=1)  # Default up vector
            )
        else:
            camera_config = default_camera
    else:
        camera_config = default_camera
    
    # Configure layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            bgcolor='lightblue',  # Light blue background
            camera=camera_config
        ),
        width=None,  # Auto-size to container
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Register this graph for camera sync 
    _created_graphs.append(graph_index)
    
    return dcc.Graph(
        id={'type': 'point-cloud-graph', 'index': graph_index},  # Use proper pattern matching ID
        figure=fig,
        style={'height': '500px', 'width': '100%'},
        config={
            'displayModeBar': True, 
            'displaylogo': False,
            'plotGlPixelRatio': 2,  # High DPI support
            'toImageButtonOptions': {'format': 'png', 'filename': f'point_cloud_{title}', 'height': 500, 'width': 700, 'scale': 1}
        }
    )


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
        html.Li(f"Total Points: {len(points_np):,}"),
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
                html.Li(f"{class_name}: {count:,} points ({percentage:.2f}%)",
                       style={'marginLeft': '20px'})
            )

        stats_items.append(html.Ul(class_list_items))

    return html.Ul(stats_items)
