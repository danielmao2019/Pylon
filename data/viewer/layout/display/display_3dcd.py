"""UI components for displaying 3D change detection dataset items."""
from typing import Dict, Optional, Any, Tuple
import torch
from dash import dcc, html
from data.viewer.utils.point_cloud import create_point_cloud_figure, get_point_cloud_stats, build_point_cloud_id
from data.viewer.utils.display_utils import (
    DisplayStyles,
    ParallelFigureCreator,
    create_standard_datapoint_layout,
    create_statistics_display
)
from data.viewer.utils.structure_validation import validate_3dcd_structure


def display_3dcd_datapoint(
    datapoint: Dict[str, Any],
    class_names: Optional[Dict[int, str]] = None,
    camera_state: Optional[Dict[str, Any]] = None,
    point_size: float = 2,
    point_opacity: float = 0.8,
    lod_type: str = "continuous",
    density_percentage: int = 100
) -> html.Div:
    """Display a 3D point cloud datapoint with all relevant information.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        class_names: Optional dictionary mapping class indices to names
        camera_state: Optional dictionary containing camera position state
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        lod_type: Type of LOD ("continuous", "discrete", or "none")
        density_percentage: Percentage of points to display when lod_type is "none" (1-100)

    Returns:
        html.Div containing the visualization
    """
    # Validate structure and inputs (includes all basic validation)
    validate_3dcd_structure(datapoint)
    
    inputs = datapoint['inputs']

    # Extract data
    points_1 = inputs['pc_1']['pos']  # First point cloud
    points_2 = inputs['pc_2']['pos']  # Second point cloud
    change_map = datapoint['labels']['change_map']
    
    # Extract RGB colors if available
    rgb_1 = inputs['pc_1'].get('rgb')
    rgb_2 = inputs['pc_2'].get('rgb')

    # Get statistics for point clouds
    stats_data = [
        get_point_cloud_stats(points_1, class_names=class_names),
        get_point_cloud_stats(points_2, class_names=class_names),
        get_point_cloud_stats(points_1, change_map, class_names=class_names)
    ]

    # Prepare figure creation tasks with proper point cloud IDs
    figure_tasks = [
        lambda: create_point_cloud_figure(
            points=points_1,
            colors=rgb_1,
            labels=None,
            title="Point Cloud 1",
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            lod_type=lod_type,
            point_cloud_id=build_point_cloud_id(datapoint, "pc_1"),
            density_percentage=density_percentage,
        ),
        lambda: create_point_cloud_figure(
            points=points_2,
            colors=rgb_2,
            labels=None,
            title="Point Cloud 2",
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            lod_type=lod_type,
            point_cloud_id=build_point_cloud_id(datapoint, "pc_2"),
            density_percentage=density_percentage,
        ),
        lambda: create_point_cloud_figure(
            points=points_2,  # Use points_2 for change map visualization
            labels=change_map,  # Keep as int64 for proper label processing
            title="Change Map",
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            lod_type=lod_type,
            point_cloud_id=build_point_cloud_id(datapoint, "change_map"),
            density_percentage=density_percentage,
        ),
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

    # Create statistics components directly from HTML returned by get_point_cloud_stats
    stats_components = [
        html.Div([
            html.H4("Point Cloud 1 Statistics:"),
            stats_data[0]  # HTML component returned by get_point_cloud_stats
        ], style=DisplayStyles.GRID_ITEM_33),

        html.Div([
            html.H4("Point Cloud 2 Statistics:"),
            stats_data[1]  # HTML component returned by get_point_cloud_stats
        ], style=DisplayStyles.GRID_ITEM_33),

        html.Div([
            html.H4("Change Statistics:"),
            stats_data[2]  # HTML component returned by get_point_cloud_stats
        ], style=DisplayStyles.GRID_ITEM_33),
    ]

    # Create complete layout
    result = create_standard_datapoint_layout(
        figure_components=fig_components,
        stats_components=stats_components,
        meta_info=datapoint.get('meta_info', {}),
        debug_outputs=datapoint.get('debug')
    )
    
    return result
