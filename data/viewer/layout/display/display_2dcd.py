"""UI components for displaying dataset items."""
from typing import Dict, List, Optional, Union, Any
import random
import numpy as np
import torch
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from data.viewer.utils.dataset_utils import format_value


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
            dcc.Graph(figure=create_2d_figure(img_1, title="Image 1"))
        ], style={'width': '33%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(figure=create_2d_figure(img_2, title="Image 2"))
        ], style={'width': '33%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(figure=create_2d_figure(change_map, title="Change Map", colorscale="Viridis"))
        ], style={'width': '33%', 'display': 'inline-block'})
    ]

    # Get statistics using helper function
    stats_components: List[html.Div] = [
        html.Div([
            html.H4("Image 1 Statistics:"),
            html.Ul([html.Li(f"{k}: {v}") for k, v in get_2d_stats(img_1).items()])
        ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),

        html.Div([
            html.H4("Image 2 Statistics:"),
            html.Ul([html.Li(f"{k}: {v}") for k, v in get_2d_stats(img_2).items()])
        ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),

        html.Div([
            html.H4("Change Statistics:"),
            html.Ul([html.Li(f"{k}: {v}") for k, v in get_2d_stats(img_1, change_map).items()])
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


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a displayable image."""
    assert isinstance(tensor, torch.Tensor), f"{tensor=}"
    if tensor.ndim == 4:
        assert tensor.shape[0] == 1, f"{tensor.shape=}"
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)

    img: np.ndarray = tensor.cpu().numpy()

    # Handle normalization with zero division check
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        # If all values are the same, return zeros
        img = np.zeros_like(img)

    if img.ndim == 2:  # Grayscale image
        return img
    elif img.ndim == 3:  # RGB image (C, H, W) -> (H, W, C)
        if img.shape[0] > 3:
            img = img[random.sample(range(img.shape[0]), 3), :, :]
        return np.transpose(img, (1, 2, 0))
    else:
        raise ValueError("Unsupported tensor shape for image conversion")


def create_2d_figure(tensor: torch.Tensor, title: str = "Image", colorscale: str = "Viridis") -> go.Figure:
    """Create a 2D image figure with standard formatting.

    Args:
        tensor: Image tensor to display
        title: Title for the figure
        colorscale: Color scale to use for the image

    Returns:
        Plotly Figure object
    """
    img: np.ndarray = tensor_to_image(tensor)

    fig = px.imshow(
        img,
        title=title,
        color_continuous_scale=colorscale
    )

    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=True,
        showlegend=False,
        height=400
    )

    return fig


def get_2d_stats(img: torch.Tensor, change_map: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Get statistical information about a 2D image.

    Args:
        img: Image tensor of shape (C, H, W)
        change_map: Optional tensor with change classes for each pixel

    Returns:
        Dictionary with image statistics
    """
    if not isinstance(img, torch.Tensor):
        return {}

    # Basic stats
    img_np: np.ndarray = img.detach().cpu().numpy()
    stats: Dict[str, Any] = {
        "Shape": f"{img_np.shape}",
        "Min Value": f"{img_np.min():.4f}",
        "Max Value": f"{img_np.max():.4f}",
        "Mean Value": f"{img_np.mean():.4f}",
        "Std Dev": f"{img_np.std():.4f}"
    }

    # Add change map statistics if provided
    if change_map is not None:
        if change_map.dim() > 2 and change_map.shape[0] > 1:
            # Multi-class change map
            change_classes: torch.Tensor = torch.argmax(change_map, dim=0)
            num_classes: int = change_map.shape[0]
            class_distribution: Dict[int, float] = {
                i: float((change_classes == i).sum()) / change_classes.numel() * 100
                for i in range(num_classes)
            }
            stats["Number of Classes"] = num_classes
            stats["Class Distribution"] = {
                f"Class {i}": f"{pct:.2f}%"
                for i, pct in class_distribution.items()
            }
        else:
            # Binary change map
            changes: torch.Tensor = change_map[0] if change_map.dim() > 2 else change_map
            percent_changed: float = float((changes > 0.5).sum()) / changes.numel() * 100
            stats["Changed Pixels"] = f"{percent_changed:.2f}%"
            stats["Change Min"] = f"{float(changes.min()):.4f}"
            stats["Change Max"] = f"{float(changes.max()):.4f}"

    return stats
