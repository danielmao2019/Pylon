"""Base class for 3D change detection datasets with built-in display methods.

This module provides the Base3DCDDataset class that inherits from BaseDataset
and includes type-specific display methods for 3D change detection datasets.
"""
from typing import Dict, Any, Optional, List
import torch
from dash import dcc, html
from data.datasets.base_dataset import BaseDataset
from data.viewer.utils.atomic_displays.point_cloud_display import create_point_cloud_display, get_point_cloud_display_stats, build_point_cloud_id
from data.viewer.utils.display_utils import (
    DisplayStyles,
    ParallelFigureCreator,
    create_standard_datapoint_layout,
    create_statistics_display
)
from data.viewer.utils.structure_validation import validate_3dcd_structure


class Base3DCDDataset(BaseDataset):
    """Base class for 3D change detection datasets.
    
    This class provides the standard INPUT_NAMES, LABEL_NAMES, and display_datapoint
    method for 3D change detection datasets. Concrete dataset classes should inherit
    from this class to automatically get appropriate display functionality.
    
    Expected data structure:
    - inputs: {'pc_1': Dict, 'pc_2': Dict, [optional] 'kdtree_1': Any, 'kdtree_2': Any}
    - labels: {'change_map': torch.Tensor}
    """
    
    INPUT_NAMES = ['pc_1', 'pc_2']
    LABEL_NAMES = ['change_map']
    
    @staticmethod
    def display_datapoint(
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> html.Div:
        """Display a 3D change detection datapoint.
        
        Args:
            datapoint: Dictionary containing inputs, labels, and meta_info from dataset
            class_labels: Optional dictionary mapping class indices to label names
            camera_state: Optional dictionary containing camera position state for 3D visualizations
            settings_3d: Optional dictionary containing 3D visualization settings
            
        Returns:
            html.Div: HTML layout for displaying this datapoint
        """
        # Validate inputs
        assert datapoint is not None, "datapoint must not be None"
        assert isinstance(datapoint, dict), f"datapoint must be dict, got {type(datapoint)}"
        
        # Handle class_labels parameter (convert to class_names format)
        class_names = None  
        if class_labels is not None:
            # Convert class_labels dict to simple int->str mapping expected by display function
            if isinstance(class_labels, dict) and len(class_labels) > 0:
                # Assume first key contains the class names list
                first_key = next(iter(class_labels))
                if isinstance(class_labels[first_key], list):
                    class_names = {i: name for i, name in enumerate(class_labels[first_key])}
        
        # Extract 3D-specific parameters with defaults
        point_size = 2.0
        point_opacity = 0.8
        lod_type = "continuous"
        density_percentage = 100
        
        # Unpack 3D settings if provided
        if settings_3d is not None:
            assert isinstance(settings_3d, dict), f"settings_3d must be dict, got {type(settings_3d)}"
            point_size = settings_3d.get('point_size', point_size)
            point_opacity = settings_3d.get('point_opacity', point_opacity)
            lod_type = settings_3d.get('lod_type', lod_type)
            density_percentage = settings_3d.get('density_percentage', density_percentage)
        
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
            get_point_cloud_display_stats(inputs['pc_1'], class_names=class_names),
            get_point_cloud_display_stats(inputs['pc_2'], class_names=class_names),
            get_point_cloud_display_stats(inputs['pc_2'], change_map, class_names=class_names)  # change_map corresponds to points_2
        ]

        # Prepare figure creation tasks with proper point cloud IDs
        figure_tasks = [
            lambda: create_point_cloud_display(
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
            lambda: create_point_cloud_display(
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
            lambda: create_point_cloud_display(
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
