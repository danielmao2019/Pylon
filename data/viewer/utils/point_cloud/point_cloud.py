"""WebGL-based point cloud visualization utilities."""
from typing import Dict, Optional, Union, Any
import os
import numpy as np
import torch
import json
import uuid
from dash import html
from data.viewer.utils.segmentation import get_color


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
    """Prepare point cloud data for WebGL rendering."""

    # Ensure points are float32 for WebGL
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

    # Calculate bounding box for camera setup
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = np.max(bbox_max - bbox_min)

    return {
        'positions': points.flatten().tolist(),  # Flatten for WebGL buffer
        'colors': colors.flatten().tolist(),
        'point_count': len(points),
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'bbox_center': bbox_center.tolist(),
        'bbox_size': float(bbox_size)
    }


def create_webgl_point_cloud_component(
    points: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Point Cloud",
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None
) -> html.Div:
    """Create a WebGL-based point cloud visualization component."""

    # Convert inputs to numpy
    points = point_cloud_to_numpy(points)
    if colors is not None:
        colors = point_cloud_to_numpy(colors)
    if labels is not None:
        labels = point_cloud_to_numpy(labels)

    # Use all points - no downsampling

    # Prepare data for WebGL
    webgl_data = prepare_point_cloud_data(points, colors, labels)

    # Default camera state
    if camera_state is None:
        camera_distance = webgl_data['bbox_size'] * 2
        camera_state = {
            'position': [camera_distance, camera_distance, camera_distance],
            'target': webgl_data['bbox_center'],
            'up': [0, 0, 1]
        }

    # Generate unique ID for this component
    component_id = f"webgl-point-cloud-{uuid.uuid4().hex[:8]}"

    # Create the WebGL component
    return html.Div([
        html.H4(title, style={'margin-bottom': '10px'}),

        # Performance info
        html.Div([
            html.Span(f"Displaying {len(points):,} points"),
            html.Span(f" • Point size: {point_size} • Opacity: {point_opacity}")
        ], style={'font-size': '12px', 'color': '#666', 'margin-bottom': '5px'}),

        # WebGL container
        html.Div(
            id=component_id,
            style={
                'width': '100%',
                'height': '500px',
                'border': '1px solid #ddd',
                'border-radius': '4px',
                'position': 'relative',
                'overflow': 'hidden'
            }
        ),

        # Load external dependencies
        html.Script(src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r158/three.min.js'),
        html.Script(src='https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js'),

        # Include our WebGL renderer and initialization helper
        html.Script(get_webgl_assets()),
        html.Script(get_init_script()),

        # Minimal inline code - just call the initialization with config
        html.Script(f"""
        initWebGLPointCloudWithConfig({{
            containerId: '{component_id}',
            pointCloudData: {json.dumps(webgl_data)},
            pointSize: {point_size},
            pointOpacity: {point_opacity},
            cameraState: {json.dumps(camera_state)}
        }});
        """),

        # Controls info
        html.Div([
            html.Strong("Controls: "),
            html.Span("Left click + drag: Rotate • Right click + drag: Pan • Scroll: Zoom • R: Reset view • +/-: Point size")
        ], style={
            'font-size': '11px',
            'color': '#888',
            'margin-top': '5px',
            'padding': '5px',
            'background-color': '#f8f9fa',
            'border-radius': '3px'
        })
    ])


def create_point_cloud_figure(
    points: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Point Cloud",
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
) -> html.Div:
    """Create a point cloud visualization.

    This is the main entry point that replaces the old Plotly-based function.
    Returns a WebGL-based visualization component.

    Args:
        points: Numpy array of shape (N, 3) containing XYZ coordinates
        colors: Optional numpy array of shape (N, 3) containing RGB color values
        labels: Optional numpy array of shape (N,) containing labels
        title: Title for the figure
        point_size: Size of the points
        point_opacity: Opacity of the points
        camera_state: Optional dictionary containing camera position state

    Returns:
        HTML Div containing WebGL point cloud visualization
    """

    return create_webgl_point_cloud_component(
        points=points,
        colors=colors,
        labels=labels,
        title=title,
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state
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


def get_webgl_assets() -> str:
    """Get the WebGL JavaScript code for point cloud rendering."""
    js_file_path = os.path.join(os.path.dirname(__file__), 'webgl_point_cloud.js')
    with open(js_file_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_init_script() -> str:
    """Get the initialization helper JavaScript code."""
    js_file_path = os.path.join(os.path.dirname(__file__), 'webgl_init.js')
    with open(js_file_path, 'r', encoding='utf-8') as f:
        return f.read()


