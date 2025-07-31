"""Base class for semantic segmentation datasets with built-in display methods.

This module provides the BaseSemsegDataset class that inherits from BaseDataset
and includes type-specific display methods for semantic segmentation datasets.
"""
from typing import Dict, Any, Optional, List, Union
import torch
from dash import dcc, html
from data.datasets.base_dataset import BaseDataset
from data.viewer.utils.image import create_image_figure, get_image_stats
from data.viewer.utils.segmentation import create_segmentation_figure, get_segmentation_stats
from data.viewer.utils.display_utils import (
    DisplayStyles,
    create_standard_datapoint_layout,
    create_statistics_display
)
from data.viewer.utils.structure_validation import validate_semseg_structure


class BaseSemsegDataset(BaseDataset):
    """Base class for semantic segmentation datasets.
    
    This class provides the standard INPUT_NAMES, LABEL_NAMES, and display_datapoint
    method for semantic segmentation datasets. Concrete dataset classes should inherit
    from this class to automatically get appropriate display functionality.
    
    Expected data structure:
    - inputs: {'image': torch.Tensor}
    - labels: {'label': torch.Tensor or Dict}  # Tensor: [H, W], Dict: instance segmentation format
    """
    
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['label']
    
    @staticmethod
    def display_datapoint(
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> html.Div:
        """Display a semantic segmentation datapoint with all relevant information.
        
        Args:
            datapoint: Dictionary containing inputs, labels, and meta_info from dataset.
                The labels should contain either:
                - A tensor of shape (H, W) with class indices
                - A dict with keys:
                    - "masks": List[torch.Tensor] of binary masks
                    - "indices": List[Any] of corresponding indices
            class_labels: Optional dictionary mapping class indices to label names (unused for semantic segmentation)
            camera_state: Optional dictionary containing camera position state (unused for semantic segmentation)
            settings_3d: Optional dictionary containing 3D visualization settings (unused for semantic segmentation)
            
        Returns:
            html.Div: HTML layout for displaying this datapoint
        """
        # Validate inputs
        assert datapoint is not None, "datapoint must not be None"
        assert isinstance(datapoint, dict), f"datapoint must be dict, got {type(datapoint)}"
        
        # Validate structure and inputs (includes all basic validation)
        validate_semseg_structure(datapoint)

        # Extract data
        image: torch.Tensor = datapoint['inputs']['image']
        seg: Union[torch.Tensor, Dict[str, Any]] = datapoint['labels']['label']

        # Create figure components
        fig_components = [
            html.Div([
                dcc.Graph(figure=create_image_figure(image, title="Image"))
            ], style=DisplayStyles.GRID_ITEM_50),

            html.Div([
                dcc.Graph(figure=create_segmentation_figure(seg, title="Segmentation Map"))
            ], style=DisplayStyles.GRID_ITEM_50)
        ]

        # Create statistics components
        stats_data = [
            get_image_stats(image),
            get_segmentation_stats(seg)
        ]
        titles = ["Image Statistics", "Segmentation Statistics"]
        stats_components = create_statistics_display(stats_data, titles, width_style="50%")

        # Create complete layout
        return create_standard_datapoint_layout(
            figure_components=fig_components,
            stats_components=stats_components,
            meta_info=datapoint.get('meta_info', {}),
            debug_outputs=datapoint.get('debug')
        )
