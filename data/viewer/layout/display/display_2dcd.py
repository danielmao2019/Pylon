"""UI components for displaying dataset items."""
from typing import Dict, Any
from dash import dcc, html
import torch
from data.viewer.utils.image import create_image_figure, get_image_stats
from data.viewer.utils.display_utils import (
    DisplayStyles,
    create_standard_datapoint_layout,
    create_statistics_display
)


def display_2dcd_datapoint(datapoint: Dict[str, Any]) -> html.Div:
    """Display a 2D image datapoint with all relevant information.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info

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
    assert datapoint['inputs'].keys() == {'img_1', 'img_2'}, f"{datapoint['inputs'].keys()=}"
    assert datapoint['labels'].keys() == {'change_map'}, f"{datapoint['labels'].keys()=}"

    # Extract data
    img_1: torch.Tensor = datapoint['inputs']['img_1']
    img_2: torch.Tensor = datapoint['inputs']['img_2']
    change_map: torch.Tensor = datapoint['labels']['change_map']

    # Create figure components
    fig_components = [
        html.Div([
            dcc.Graph(figure=create_image_figure(img_1, title="Image 1"))
        ], style=DisplayStyles.GRID_ITEM_33),

        html.Div([
            dcc.Graph(figure=create_image_figure(img_2, title="Image 2"))
        ], style=DisplayStyles.GRID_ITEM_33),

        html.Div([
            dcc.Graph(figure=create_image_figure(change_map, title="Change Map", colorscale="Viridis"))
        ], style=DisplayStyles.GRID_ITEM_33)
    ]

    # Create statistics components
    stats_data = [
        get_image_stats(img_1),
        get_image_stats(img_2),
        get_image_stats(img_1, change_map)
    ]
    titles = ["Image 1 Statistics", "Image 2 Statistics", "Change Statistics"]
    stats_components = create_statistics_display(stats_data, titles, width_style="33%")

    # Create complete layout
    return create_standard_datapoint_layout(
        figure_components=fig_components,
        stats_components=stats_components,
        meta_info=datapoint.get('meta_info', {}),
        debug_outputs=datapoint.get('debug')
    )
