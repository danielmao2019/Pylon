"""Base class for 2D change detection datasets with built-in display methods.

This module provides the Base2DCDDataset class that inherits from BaseDataset
and includes type-specific display methods for 2D change detection datasets.
"""
from typing import Dict, Any, Optional, List
from dash import dcc, html
import torch
from data.datasets.base_dataset import BaseDataset
from data.viewer.utils.atomic_displays.image_display import create_image_display, get_image_display_stats
from data.viewer.utils.atomic_displays.segmentation_display import create_segmentation_display, get_segmentation_display_stats
from data.viewer.utils.display_utils import (
    DisplayStyles,
    create_standard_datapoint_layout,
    create_statistics_display
)
from data.viewer.utils.structure_validation import validate_2dcd_structure


class Base2DCDDataset(BaseDataset):
    """Base class for 2D change detection datasets.
    
    This class provides the standard INPUT_NAMES, LABEL_NAMES, and display_datapoint
    method for 2D change detection datasets. Concrete dataset classes should inherit
    from this class to automatically get appropriate display functionality.
    
    Expected data structure:
    - inputs: {'img_1': torch.Tensor, 'img_2': torch.Tensor}
    - labels: {'change_map': torch.Tensor}
    """
    
    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']
    
    @staticmethod
    def display_datapoint(
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> html.Div:
        """Display a 2D change detection datapoint.
        
        Args:
            datapoint: Dictionary containing inputs, labels, and meta_info from dataset
            class_labels: Optional dictionary mapping class indices to label names (unused for 2DCD)
            camera_state: Optional dictionary containing camera position state (unused for 2DCD)
            settings_3d: Optional dictionary containing 3D visualization settings (unused for 2DCD)
            
        Returns:
            html.Div: HTML layout for displaying this datapoint
        """
        # Validate inputs
        assert datapoint is not None, "datapoint must not be None"
        assert isinstance(datapoint, dict), f"datapoint must be dict, got {type(datapoint)}"
        
        # Validate structure and inputs (includes all basic validation)
        validate_2dcd_structure(datapoint)

        # Extract data
        img_1: torch.Tensor = datapoint['inputs']['img_1']
        img_2: torch.Tensor = datapoint['inputs']['img_2']
        change_map: torch.Tensor = datapoint['labels']['change_map']

        # Create figure components
        fig_components = [
            html.Div([
                dcc.Graph(figure=create_image_display(img_1, title="Image 1"))
            ], style=DisplayStyles.GRID_ITEM_33),

            html.Div([
                dcc.Graph(figure=create_image_display(img_2, title="Image 2"))
            ], style=DisplayStyles.GRID_ITEM_33),

            html.Div([
                dcc.Graph(figure=create_segmentation_display(change_map, title="Change Map", class_labels=class_labels))
            ], style=DisplayStyles.GRID_ITEM_33)
        ]

        # Create statistics components
        stats_data = [
            get_image_display_stats(img_1),
            get_image_display_stats(img_2),
            get_segmentation_display_stats(change_map)
        ]
        titles = ["Image 1 Statistics", "Image 2 Statistics", "Change Map Statistics"]
        stats_components = create_statistics_display(stats_data, titles, width_style="33%")

        # Create complete layout
        return create_standard_datapoint_layout(
            figure_components=fig_components,
            stats_components=stats_components,
            meta_info=datapoint.get('meta_info', {}),
            debug_outputs=datapoint.get('debug')
        )
