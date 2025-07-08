"""UI components for displaying dataset items."""
from typing import Dict, Optional, Any
import torch
from dash import dcc, html
from data.viewer.utils.point_cloud import create_point_cloud_figure, get_point_cloud_stats
from data.viewer.utils.display_utils import (
    DisplayStyles,
    ParallelFigureCreator,
    create_standard_datapoint_layout,
    create_statistics_display
)


def display_3dcd_datapoint(
    datapoint: Dict[str, Any],
    class_names: Optional[Dict[int, str]] = None,
    camera_state: Optional[Dict[str, Any]] = None,
    point_size: float = 2,
    point_opacity: float = 0.8,
    lod_type: str = "continuous"
) -> html.Div:
    """Display a 3D point cloud datapoint with all relevant information.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        class_names: Optional dictionary mapping class indices to names
        camera_state: Optional dictionary containing camera position state
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        lod_type: Type of LOD ("continuous", "discrete", or "none")

    Returns:
        html.Div containing the visualization
    """
    # Validate inputs
    inputs = datapoint['inputs']
    assert 'pc_1' in inputs and 'pc_2' in inputs, "Point cloud 1 (pc_1) and point cloud 2 (pc_2) must be present in the inputs"
    assert isinstance(inputs['pc_1'], dict) and isinstance(inputs['pc_2'], dict), "Point clouds must be dictionaries"

    # Extract data
    points_1 = inputs['pc_1']['pos']  # First point cloud
    points_2 = inputs['pc_2']['pos']  # Second point cloud
    change_map = datapoint['labels']['change_map']

    # Get statistics for point clouds
    stats_data = [
        get_point_cloud_stats(points_1, class_names=class_names),
        get_point_cloud_stats(points_2, class_names=class_names),
        get_point_cloud_stats(points_1, change_map, class_names=class_names)
    ]

    # Prepare figure creation tasks
    def create_figure_task(points, labels, title, pc_id):
        return lambda: create_point_cloud_figure(
            points=points,
            labels=labels,
            title=title,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            lod_type=lod_type,
            point_cloud_id=pc_id,
        )

    # Prepare data for figures
    points_list = [points_1, points_2]
    labels_list = [None, None]
    
    # For change map visualization, use pc_1 with colors from change_map
    if change_map is not None:
        points_list.append(points_2)
        labels_list.append(change_map.float())  # Convert to float for proper coloring

    titles = ["Point Cloud 1", "Point Cloud 2", "Change Map"]

    # Create figure tasks
    figure_tasks = [
        create_figure_task(points, labels, title, f"3dcd_{idx}")
        for idx, (points, labels, title) in enumerate(zip(points_list, labels_list, titles))
    ]

    # Create figures in parallel
    figure_creator = ParallelFigureCreator(max_workers=3, enable_timing=True)
    figures = figure_creator.create_figures_parallel(figure_tasks, "3DCD Display")

    # Create figure components
    fig_components = [
        html.Div([
            dcc.Graph(figure=figures[0], id={'type': 'point-cloud-graph', 'index': 0})
        ], style=DisplayStyles.GRID_ITEM_33),

        html.Div([
            dcc.Graph(figure=figures[1], id={'type': 'point-cloud-graph', 'index': 1})
        ], style=DisplayStyles.GRID_ITEM_33),

        html.Div([
            dcc.Graph(figure=figures[2] if len(figures) > 2 else {},
                     id={'type': 'point-cloud-graph', 'index': 2})
        ], style=DisplayStyles.GRID_ITEM_33),
    ]

    # Create statistics components
    # Convert HTML components to dictionary format for create_statistics_display
    stats_dict_data = []
    for stats in stats_data:
        if hasattr(stats, 'children'):
            # Convert HTML.Ul to dict format for consistency
            stats_dict = {}
            for child in stats.children:
                if hasattr(child, 'children'):
                    stats_dict[f"Stat {len(stats_dict)}"] = str(child.children)
            stats_dict_data.append(stats_dict)
        else:
            stats_dict_data.append(stats)

    titles = ["Point Cloud 1 Statistics", "Point Cloud 2 Statistics", "Change Statistics"]
    
    # Create custom statistics display since we have HTML components
    stats_components = [
        html.Div([
            html.H4("Point Cloud 1 Statistics:"),
            html.Div(stats_data[0])
        ], style=DisplayStyles.STATS_CONTAINER),

        html.Div([
            html.H4("Point Cloud 2 Statistics:"),
            html.Div(stats_data[1])
        ], style=DisplayStyles.STATS_CONTAINER),

        html.Div([
            html.H4("Change Statistics:"),
            html.Div(stats_data[2])
        ], style=DisplayStyles.STATS_CONTAINER),
    ]

    # Create complete layout
    result = create_standard_datapoint_layout(
        figure_components=fig_components,
        stats_components=stats_components,
        meta_info=datapoint.get('meta_info', {}),
        debug_outputs=datapoint.get('debug')
    )
    
    return result
