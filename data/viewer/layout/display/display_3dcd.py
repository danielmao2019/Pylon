"""UI components for displaying dataset items."""
from typing import Dict, Optional, Any
from dash import dcc, html
from data.viewer.utils.dataset_utils import format_value
from data.viewer.utils.point_cloud import create_point_cloud_figure, get_point_cloud_stats


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
    pc_1_stats_children = get_point_cloud_stats(pc_1, class_names=class_names)
    pc_2_stats_children = get_point_cloud_stats(pc_2, class_names=class_names)
    change_stats_children = get_point_cloud_stats(pc_1, change_map, class_names=class_names)

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
        fig = create_point_cloud_figure(
            pc,
            colors=color,
            title=title,
            point_size=point_size,
            point_opacity=point_opacity,
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
