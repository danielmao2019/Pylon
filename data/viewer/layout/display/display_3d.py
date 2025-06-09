"""UI components for displaying dataset items."""
from typing import Dict, Optional, Union, Any
import numpy as np
import torch
from dash import dcc, html
import plotly.graph_objects as go
from data.viewer.utils.dataset_utils import format_value


def display_3dcd_datapoint(
    datapoint: Dict[str, Any],
    point_size: float = 2,
    point_opacity: float = 0.8,
    class_names: Optional[Dict[int, str]] = None,
    camera_state: Optional[Dict[str, Any]] = None
) -> html.Div:
    """Display a 3D point cloud datapoint with all relevant information.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        class_names: Optional dictionary mapping class indices to names
        camera_state: Optional dictionary containing camera position state

    Returns:
        html.Div containing the visualization
    """
    # Check if the inputs have the expected structure
    inputs = datapoint['inputs']
    assert 'pc_1' in inputs and 'pc_2' in inputs, "Point cloud 1 (pc_1) and point cloud 2 (pc_2) must be present in the inputs"
    assert isinstance(inputs['pc_1'], dict) and isinstance(inputs['pc_2'], dict), "Point clouds must be dictionaries"

    pc_1 = inputs['pc_1']['pos']  # First point cloud
    pc_2 = inputs['pc_2']['pos']  # Second point cloud
    change_map = datapoint['labels']['change_map']

    # Get stats for point clouds
    pc_1_stats_children = get_3d_stats(pc_1, class_names=class_names)
    pc_2_stats_children = get_3d_stats(pc_2, class_names=class_names)
    change_stats_children = get_3d_stats(pc_1, change_map, class_names=class_names)

    # Create figures for point clouds
    point_clouds = [pc_1, pc_2]
    colors = [None, None]

    # For change map visualization, we'll use pc_1 with colors from change_map
    if change_map is not None:
        point_clouds.append(pc_1)
        colors.append(change_map.float())  # Convert to float for proper coloring

    titles = ["Point Cloud 1", "Point Cloud 2", "Change Map"]

    # Create figures
    figures = []
    for i, (pc, color, title) in enumerate(zip(point_clouds, colors, titles)):
        fig = create_3d_figure(
            pc,
            colors=color,
            title=title,
            point_size=point_size,
            opacity=point_opacity,
            colorscale='Reds' if i == 2 else 'Viridis',
            camera_state=camera_state
        )
        figures.append(fig)

    # Extract metadata
    meta_info = datapoint.get('meta_info', {})
    meta_display = []
    if meta_info:
        meta_display = [
            html.H4("Metadata:"),
            html.Pre(
                format_value(meta_info),
                style={
                    'background-color': '#f0f0f0', 'padding': '10px', 'max-height': '200px',
                    'overflow-y': 'auto', 'border-radius': '5px',
                })
        ]

    # Compile the complete display
    return html.Div([
        # Point cloud displays
        html.Div([
            html.Div([
                dcc.Graph(figure=figures[0], id={'type': 'point-cloud-graph', 'index': 0})
            ], style={'width': '33%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(figure=figures[1], id={'type': 'point-cloud-graph', 'index': 1})
            ], style={'width': '33%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(figure=figures[2] if len(figures) > 2 else {},
                         id={'type': 'point-cloud-graph', 'index': 2})
            ], style={'width': '33%', 'display': 'inline-block'}),
        ]),

        # Info section
        html.Div([
            # Point cloud statistics
            html.Div([
                html.Div([
                    html.H4("Point Cloud 1 Statistics:"),
                    html.Div(pc_1_stats_children)
                ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),

                html.Div([
                    html.H4("Point Cloud 2 Statistics:"),
                    html.Div(pc_2_stats_children)
                ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),

                html.Div([
                    html.H4("Change Statistics:"),
                    html.Div(change_stats_children)
                ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
            ]),

            # Metadata
            html.Div(meta_display, style={'margin-top': '20px'})
        ], style={'margin-top': '20px'})
    ])


def tensor_to_point_cloud(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a PyTorch tensor to a displayable point cloud."""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor


def get_3d_stats(
    pc: torch.Tensor,
    change_map: Optional[torch.Tensor] = None,
    class_names: Optional[Dict[int, str]] = None
) -> html.Ul:
    """Get statistical information about a point cloud.

    Args:
        pc: Point cloud tensor of shape (N, 3+)
        change_map: Optional tensor with change classes for each point
        class_names: Optional dictionary mapping class IDs to class names

    Returns:
        List of html components with point cloud statistics
    """
    # Basic stats
    pc_np = pc.detach().cpu().numpy()
    stats_items = [
        html.Li(f"Total Points: {len(pc_np)}"),
        html.Li(f"Dimensions: {pc_np.shape[1]}"),
        html.Li(f"X Range: [{pc_np[:, 0].min():.2f}, {pc_np[:, 0].max():.2f}]"),
        html.Li(f"Y Range: [{pc_np[:, 1].min():.2f}, {pc_np[:, 1].max():.2f}]"),
        html.Li(f"Z Range: [{pc_np[:, 2].min():.2f}, {pc_np[:, 2].max():.2f}]"),
        html.Li(f"Center: [{pc_np[:, 0].mean():.2f}, {pc_np[:, 1].mean():.2f}, {pc_np[:, 2].mean():.2f}]")
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


def create_3d_figure(
    pc_data: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Point Cloud",
    colorscale: str = 'Viridis',
    point_size: float = 2,
    opacity: float = 0.8,
    colorbar_title: str = "Class",
    camera_state: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """Create a 3D point cloud visualization figure.

    Args:
        pc_data: Numpy array of shape (N, 3) containing XYZ coordinates
        colors: Optional numpy array of shape (N,) containing color values
        title: Title for the figure
        colorscale: Colorscale to use for the point cloud
        point_size: Size of the points
        opacity: Opacity of the points
        colorbar_title: Title for the colorbar
        camera_state: Optional dictionary containing camera position state

    Returns:
        Plotly Figure object
    """
    # Convert input data to numpy arrays
    pc_data = tensor_to_point_cloud(pc_data)
    if colors is not None:
        colors = tensor_to_point_cloud(colors)

    # Subsample large point clouds if necessary for better performance
    max_points = 100000  # Adjust based on performance needs
    if pc_data.shape[0] > max_points:
        indices = np.random.choice(pc_data.shape[0], max_points, replace=False)
        pc_data = pc_data[indices]
        if colors is not None:
            colors = colors[indices]

    # Create figure
    fig = go.Figure()

    # Add point cloud
    if colors is not None:
        fig.add_trace(go.Scatter3d(
            x=pc_data[:, 0],
            y=pc_data[:, 1],
            z=pc_data[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=colors,
                colorscale=colorscale,
                opacity=opacity,
                colorbar=dict(
                    title=colorbar_title,
                    thickness=15,
                    len=0.6,
                    x=1.02,
                    xanchor="left",
                    xpad=10
                )
            ),
            text=[f"Point {i}<br>Value: {c}" for i, c in enumerate(colors)],
            hoverinfo='text'
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=pc_data[:, 0],
            y=pc_data[:, 1],
            z=pc_data[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color='steelblue',
                opacity=opacity
            ),
            text=[f"Point {i}" for i in range(len(pc_data))],
            hoverinfo='text'
        ))

    # Calculate bounding box
    x_range = [pc_data[:, 0].min(), pc_data[:, 0].max()]
    y_range = [pc_data[:, 1].min(), pc_data[:, 1].max()]
    z_range = [pc_data[:, 2].min(), pc_data[:, 2].max()]

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
