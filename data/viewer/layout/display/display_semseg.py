"""UI components for displaying semantic segmentation dataset items."""
from typing import Dict, List, Optional, Union, Any
import random
import numpy as np
import torch
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from data.viewer.utils.dataset_utils import format_value


def generate_unique_colors(num_classes: int) -> List[str]:
    """Generate a list of unique colors for class visualization.

    Args:
        num_classes: Number of classes to generate colors for

    Returns:
        List of hex color codes
    """
    # Use a fixed seed for reproducibility
    random.seed(42)

    # Generate colors using HSV color space for better distribution
    colors = []
    for i in range(num_classes):
        # Use golden ratio to get well-distributed hues
        hue = (i * 0.618033988749895) % 1.0
        # Use high saturation and value for better visibility
        saturation = 0.8
        value = 0.9

        # Convert HSV to RGB
        h = hue
        s = saturation
        v = value

        if s == 0.0:
            r, g, b = v, v, v
        else:
            h *= 6.0
            i = int(h)
            f = h - i
            p = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))

            if i == 0:
                r, g, b = v, t, p
            elif i == 1:
                r, g, b = q, v, p
            elif i == 2:
                r, g, b = p, v, t
            elif i == 3:
                r, g, b = p, q, v
            elif i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q

        # Convert to hex
        hex_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        colors.append(hex_color)

    return colors


def display_semseg_datapoint(
    datapoint: Dict[str, Any],
    class_names: Optional[Dict[int, str]] = None
) -> html.Div:
    """Display a semantic segmentation datapoint with all relevant information.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        class_names: Optional dictionary mapping class indices to names

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
    seg_map: torch.Tensor = datapoint['labels']['label']

    # Get number of classes from segmentation map
    num_classes = int(seg_map.max().item()) + 1
    colors = generate_unique_colors(num_classes)

    # Create the figures
    fig_components: List[html.Div] = [
        html.Div([
            dcc.Graph(figure=create_image_figure(image, title="Image"))
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(figure=create_semseg_figure(seg_map, colors, class_names, title="Segmentation Map"))
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
            html.Ul([html.Li(f"{k}: {v}") for k, v in get_semseg_stats(seg_map, class_names).items()])
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


def create_image_figure(tensor: torch.Tensor, title: str = "Image") -> go.Figure:
    """Create an image figure with standard formatting."""
    img: np.ndarray = tensor_to_image(tensor)

    fig = px.imshow(
        img,
        title=title
    )

    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        height=400
    )

    return fig


def create_semseg_figure(
    seg_map: torch.Tensor,
    colors: List[str],
    class_names: Optional[Dict[int, str]] = None,
    title: str = "Segmentation Map"
) -> go.Figure:
    """Create a segmentation map figure with class colors."""
    # Convert segmentation map to numpy
    seg_np = seg_map.cpu().numpy()

    # Create a colored segmentation map
    colored_map = np.zeros((*seg_np.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(colors):
        mask = seg_np == class_idx
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        colored_map[mask] = [r, g, b]

    fig = px.imshow(
        colored_map,
        title=title
    )

    # Add legend for classes
    if class_names:
        for class_idx, class_name in class_names.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=colors[class_idx]),
                name=class_name,
                showlegend=True
            ))

    fig.update_layout(
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        height=400
    )

    return fig


def get_image_stats(img: torch.Tensor) -> Dict[str, Any]:
    """Get statistical information about an image."""
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
    return stats


def get_semseg_stats(
    seg_map: torch.Tensor,
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """Get statistical information about a segmentation map."""
    if not isinstance(seg_map, torch.Tensor):
        return {}

    # Basic stats
    seg_np = seg_map.cpu().numpy()
    num_classes = int(seg_map.max().item()) + 1

    stats = {
        "Shape": f"{seg_np.shape}",
        "Number of Classes": num_classes,
        "Class Distribution": {}
    }

    # Calculate class distribution
    for class_idx in range(num_classes):
        class_name = class_names.get(class_idx, f"Class {class_idx}") if class_names else f"Class {class_idx}"
        class_pixels = (seg_np == class_idx).sum()
        class_percentage = (class_pixels / seg_np.size) * 100
        stats["Class Distribution"][class_name] = f"{class_percentage:.2f}%"

    return stats
