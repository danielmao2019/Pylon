"""UI components for displaying semantic segmentation dataset items."""
from typing import Dict, List, Union, Any
import torch
from dash import dcc, html
from data.viewer.utils.dataset_utils import format_value
from data.viewer.utils.image import create_image_figure, get_image_stats
from data.viewer.utils.segmentation import create_segmentation_figure, get_segmentation_stats


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
    assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}, f"{datapoint.keys()=}"
    assert 'image' in datapoint['inputs'], f"{datapoint['inputs'].keys()=}"
    assert 'label' in datapoint['labels'], f"{datapoint['labels'].keys()=}"

    # Get the image and segmentation map
    image: torch.Tensor = datapoint['inputs']['image']
    seg: Union[torch.Tensor, Dict[str, Any]] = datapoint['labels']['label']

    # Create the figures
    fig_components: List[html.Div] = [
        html.Div([
            dcc.Graph(figure=create_image_figure(image, title="Image"))
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(figure=create_segmentation_figure(seg, title="Segmentation Map"))
        ], style={'width': '50%', 'display': 'inline-block'})
    ]

    # Get statistics
    stats_components: List[html.Div] = [
        html.Div([
            html.H4("Image Statistics:"),
            html.Ul([html.Li(f"{k}: {v}") for k, v in get_image_stats(image).items()])
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),

        html.Div([
            html.H4("Segmentation Statistics:"),
            html.Ul([html.Li(f"{k}: {v}") for k, v in get_segmentation_stats(seg).items()])
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'})
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
