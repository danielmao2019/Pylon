"""UI components for displaying dataset items."""
from dash import dcc, html
import plotly.express as px
import torch
import numpy as np
import random
from data.viewer.utils.dataset_utils import format_value


def display_2d_datapoint(datapoint):
    """
    Display a 2D image datapoint with all relevant information.
    
    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        
    Returns:
        html.Div containing the visualization
    """
    # Check if the inputs have the expected structure
    img_1 = datapoint['inputs'].get('img_1')
    img_2 = datapoint['inputs'].get('img_2')
    change_map = datapoint['labels'].get('change_map')
    
    # Verify that all required data is present and has the correct type
    error_messages = []
    
    if img_1 is None:
        error_messages.append("Image 1 (img_1) is missing")
    elif not isinstance(img_1, torch.Tensor):
        error_messages.append(f"Image 1 (img_1) has unexpected type: {type(img_1).__name__}")
        
    if img_2 is None:
        error_messages.append("Image 2 (img_2) is missing")
    elif not isinstance(img_2, torch.Tensor):
        error_messages.append(f"Image 2 (img_2) has unexpected type: {type(img_2).__name__}")
        
    if change_map is None:
        error_messages.append("Change map is missing")
    elif not isinstance(change_map, torch.Tensor):
        error_messages.append(f"Change map has unexpected type: {type(change_map).__name__}")
    
    # If any errors were found, display them
    if error_messages:
        return html.Div([
            html.H3("Error Displaying 2D Image Data", style={'color': 'red'}),
            html.P("The dataset structure doesn't match the expected format:"),
            html.Ul([html.Li(msg) for msg in error_messages]),
            html.P("Datapoint structure:"),
            html.Div([
                html.P(f"Inputs keys: {list(datapoint['inputs'].keys())}"),
                html.P(f"Labels keys: {list(datapoint['labels'].keys())}"),
                html.Pre(f"Meta info: {format_value(datapoint.get('meta_info', {}))}")
            ], style={'background-color': '#f0f0f0', 'padding': '10px', 'border-radius': '5px'})
        ])

    # Convert tensors to displayable images
    try:
        img_1_display = tensor_to_image(img_1)
        img_2_display = tensor_to_image(img_2)
        change_map_display = tensor_to_image(change_map)
        
        # Create the figures using helper function
        fig_img_1 = create_2d_figure(img_1_display, title="Image 1")
        fig_img_2 = create_2d_figure(img_2_display, title="Image 2")
        fig_change = create_2d_figure(change_map_display, title="Change Map", colorscale="Viridis")
        
        # Get statistics using helper function
        img_1_stats = get_2d_stats(img_1)
        img_2_stats = get_2d_stats(img_2)
        change_stats = get_2d_stats(img_1, change_map)
        
        # Extract metadata
        meta_info = datapoint.get('meta_info', {})
        meta_display = []
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
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_img_1)
                ], style={'width': '33%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(figure=fig_img_2)
                ], style={'width': '33%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(figure=fig_change)
                ], style={'width': '33%', 'display': 'inline-block'}),
            ]),
            
            # Info section
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("Image 1 Statistics:"),
                        html.Ul([html.Li(f"{k}: {v}") for k, v in img_1_stats.items()])
                    ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    
                    html.Div([
                        html.H4("Image 2 Statistics:"),
                        html.Ul([html.Li(f"{k}: {v}") for k, v in img_2_stats.items()])
                    ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    
                    html.Div([
                        html.H4("Change Statistics:"),
                        html.Ul([html.Li(f"{k}: {v}") for k, v in change_stats.items()])
                    ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
                ]),
                html.Div(meta_display, style={'margin-top': '20px'})
            ], style={'margin-top': '20px'})
        ])
    except Exception as e:
        return html.Div([
            html.H3("Error Processing Images", style={'color': 'red'}),
            html.P(f"An error occurred while processing the images: {str(e)}"),
            html.P("Datapoint structure:"),
            html.P(f"Inputs keys: {list(datapoint['inputs'].keys())}"),
            html.P(f"Labels keys: {list(datapoint['labels'].keys())}"),
            html.P(f"Input 1 shape: {img_1.shape if img_1 is not None else 'None'}"),
            html.P(f"Input 2 shape: {img_2.shape if img_2 is not None else 'None'}"),
            html.P(f"Change map shape: {change_map.shape if change_map is not None else 'None'}")
        ])


def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a displayable image."""
    img = tensor.cpu().numpy()
    img = (img-img.min())/(img.max()-img.min())
    if img.ndim == 2:  # Grayscale image
        return img
    elif img.ndim == 3:  # RGB image (C, H, W) -> (H, W, C)
        if img.shape[0] > 3:
            img = img[random.sample(range(img.shape[0]), 3), :, :]
        return np.transpose(img, (1, 2, 0))
    else:
        raise ValueError("Unsupported tensor shape for image conversion")


def get_2d_stats(img, change_map=None):
    """Get statistical information about a 2D image.

    Args:
        img: Image tensor of shape (C, H, W)
        change_map: Optional tensor with change classes for each pixel

    Returns:
        Dictionary with image statistics
    """
    if not isinstance(img, torch.Tensor):
        return {}

    try:
        # Basic stats
        img_np = img.detach().cpu().numpy()
        stats = {
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
                change_classes = torch.argmax(change_map, dim=0)
                num_classes = change_map.shape[0]
                class_distribution = {
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
                changes = change_map[0] if change_map.dim() > 2 else change_map
                percent_changed = float((changes > 0.5).sum()) / changes.numel() * 100
                stats["Changed Pixels"] = f"{percent_changed:.2f}%"
                stats["Change Min"] = f"{float(changes.min()):.4f}"
                stats["Change Max"] = f"{float(changes.max()):.4f}"

        return stats

    except Exception as e:
        return {"Error": str(e)}


def create_2d_figure(img, title="Image", colorscale="Viridis"):
    """Create a 2D image figure with standard formatting.
    
    Args:
        img: Image array to display
        title: Title for the figure
        colorscale: Color scale to use for the image
        
    Returns:
        Plotly Figure object
    """
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
