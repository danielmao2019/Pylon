"""UI components for displaying dataset items."""
from typing import Dict, List, Union, Any
import torch
from dash import dcc, html
from data.viewer.utils.dataset_utils import format_value
from data.viewer.utils.image import create_image_figure, get_image_stats


def display_2dcd_datapoint(datapoint: Dict[str, Any]) -> html.Div:
    """
    Display a 2D image datapoint with all relevant information.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info

    Returns:
        html.Div containing the visualization
    """
    assert datapoint is not None, f"{datapoint=}"
    assert isinstance(datapoint, dict), f"{datapoint=}"
    assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}, f"{datapoint.keys()=}"
    assert datapoint['inputs'].keys() == {'img_1', 'img_2'}, f"{datapoint['inputs'].keys()=}"
    assert datapoint['labels'].keys() == {'change_map'}, f"{datapoint['labels'].keys()=}"

    # Check if the inputs have the expected structure
    img_1: torch.Tensor = datapoint['inputs']['img_1']
    img_2: torch.Tensor = datapoint['inputs']['img_2']
    change_map: torch.Tensor = datapoint['labels']['change_map']

    # Create the figures using helper function
    fig_components: List[html.Div] = [
        html.Div([
            dcc.Graph(figure=create_image_figure(img_1, title="Image 1"))
        ], style={'width': '33%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(figure=create_image_figure(img_2, title="Image 2"))
        ], style={'width': '33%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(figure=create_image_figure(change_map, title="Change Map", colorscale="Viridis"))
        ], style={'width': '33%', 'display': 'inline-block'})
    ]

    # Get statistics using helper function
    stats_components: List[html.Div] = [
        html.Div([
            html.H4("Image 1 Statistics:"),
            html.Ul([html.Li(f"{k}: {v}") for k, v in get_image_stats(img_1).items()])
        ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),

        html.Div([
            html.H4("Image 2 Statistics:"),
            html.Ul([html.Li(f"{k}: {v}") for k, v in get_image_stats(img_2).items()])
        ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),

        html.Div([
            html.H4("Change Statistics:"),
            html.Ul([html.Li(f"{k}: {v}") for k, v in get_image_stats(img_1, change_map).items()])
        ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'})
    ]

    # Extract metadata
    meta_info: Dict[str, Any] = datapoint.get('meta_info', {})
    meta_display: List[Union[html.H4, html.Pre]] = []
    if meta_info:
        meta_display = [
            html.H4("Metadata:"),
            html.Pre(format_value(meta_info),
                    style={'background-color': '#f0f0f0', 'padding': '10px', 'max-height': '200px',
                            'overflow-y': 'auto', 'border-radius': '5px'})
        ]

    # Compile the complete display
    return html.Div([
        # Image displays
        html.Div(fig_components),

        # Info section
        html.Div([
            html.Div(stats_components),
            html.Div(meta_display, style={'margin-top': '20px'})
        ], style={'margin-top': '20px'})
    ])
