"""UI components for displaying semantic segmentation dataset items."""
from typing import Dict, Union, Any
import torch
from dash import dcc, html
from data.viewer.utils.image import create_image_figure, get_image_stats
from data.viewer.utils.segmentation import create_segmentation_figure, get_segmentation_stats
from data.viewer.utils.display_utils import (
    DisplayStyles,
    create_standard_datapoint_layout,
    create_statistics_display
)


def display_semseg_datapoint(datapoint: Dict[str, Any]) -> html.Div:
    """Display a semantic segmentation datapoint with all relevant information.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info.
            The labels should contain either:
            - A tensor of shape (H, W) with class indices
            - A dict with keys:
                - "masks": List[torch.Tensor] of binary masks
                - "indices": List[Any] of corresponding indices

    Returns:
        html.Div containing the visualization
    """
    assert datapoint is not None, f"{datapoint=}"
    assert isinstance(datapoint, dict), f"{datapoint=}"
    expected_keys = {'inputs', 'labels', 'meta_info'}
    # Debug outputs are optional
    if 'debug' in datapoint:
        expected_keys.add('debug')
    assert datapoint.keys() == expected_keys, f"{datapoint.keys()=}"
    assert 'image' in datapoint['inputs'], f"{datapoint['inputs'].keys()=}"
    assert 'label' in datapoint['labels'], f"{datapoint['labels'].keys()=}"

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
