"""UI components for displaying dataset items."""
from typing import Dict, Optional, Any
from dash import dcc, html
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    points_1 = inputs['pc_1']['pos']  # First point cloud
    points_2 = inputs['pc_2']['pos']  # Second point cloud
    change_map = datapoint['labels']['change_map']

    # Get stats for point clouds
    pc_1_stats_children = get_point_cloud_stats(points_1, class_names=class_names)
    pc_2_stats_children = get_point_cloud_stats(points_2, class_names=class_names)
    change_stats_children = get_point_cloud_stats(points_1, change_map, class_names=class_names)

    # Create figures for point clouds
    points_list = [points_1, points_2]
    labels_list = [None, None]

    # For change map visualization, we'll use pc_1 with colors from change_map
    if change_map is not None:
        points_list.append(points_2)
        labels_list.append(change_map.float())  # Convert to float for proper coloring

    titles = ["Point Cloud 1", "Point Cloud 2", "Change Map"]

    # Create figures in parallel for better performance
    def create_figure(points, labels, title):
        return create_point_cloud_figure(
            points=points,
            labels=labels,
            title=title,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
        )

    # Prepare figure creation tasks
    figure_tasks = [
        (points, labels, title) 
        for points, labels, title in zip(points_list, labels_list, titles)
    ]

    figures = [None] * len(figure_tasks)  # Pre-allocate list to maintain order
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(create_figure, points, labels, title): idx 
            for idx, (points, labels, title) in enumerate(figure_tasks)
        }
        
        # Collect results in order
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            figures[idx] = future.result()

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
