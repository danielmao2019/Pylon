"""UI components for displaying 2D image data in the runners viewer."""
from dash import dcc, html
import torch
from typing import Dict, List, Optional, Union, Any
from runners.viewer.utils import tensor_to_image, create_image_figure, format_value, get_image_stats


def create_2d_display(data: Dict[str, Any], title: str = "") -> html.Div:
    """Create a display component for 2D image data.
    
    Args:
        data: Dictionary containing image data and metadata
        title: Optional title for the display
        
    Returns:
        Dash HTML Div component containing the display
    """
    try:
        # Get image data and stats
        image = data.get("image")
        if image is None:
            raise ValueError("No image data provided")
            
        image_stats = get_image_stats(image)
        
        # Convert image to displayable format
        img_array = tensor_to_image(image)
        
        # Create figure
        fig = create_image_figure(img_array, title=title)
        
        # Create stats display
        stats_items = []
        for key, value in image_stats.items():
            formatted_value = format_value(value)
            stats_items.append(
                html.Div([
                    html.Strong(f"{key}: "),
                    html.Span(formatted_value)
                ])
            )
        
        # Create display component
        return html.Div([
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False}
            ),
            html.Div(
                stats_items,
                style={"marginTop": "10px"}
            )
        ])
        
    except Exception as e:
        # Return error message on failure
        return html.Div([
            html.P(f"Error displaying image: {str(e)}",
                  style={"color": "red"})
        ])
