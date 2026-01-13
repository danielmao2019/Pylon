"""Utilities for displaying debug outputs in the data viewer."""
from typing import Dict, Any, List, Union
from dash import html, dcc
import torch
import numpy as np
from data.viewer.utils.dataset_utils import format_value
from data.viewer.utils.atomic_displays.image_display import create_image_display


def display_debug_outputs(debug_outputs: Dict[str, Any]) -> html.Div:
    """Display debug outputs from debuggers.

    Args:
        debug_outputs: Dictionary mapping debugger names to their outputs

    Returns:
        html.Div containing the debug visualization
    """
    if not debug_outputs:
        return html.Div([html.P("No debug outputs available")])

    debug_components = []

    for debugger_name, output in debug_outputs.items():
        debugger_components = [html.H5(f"Debugger: {debugger_name}")]

        if isinstance(output, dict):
            # Display structured debug output
            for key, value in output.items():
                debugger_components.append(_display_debug_item(key, value))
        else:
            # Display raw debug output
            debugger_components.append(_display_debug_item("output", output))

        debug_components.append(
            html.Div(
                debugger_components,
                style={
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'margin': '10px 0',
                    'border-radius': '5px'
                }
            )
        )

    return html.Div([
        html.H4("Debug Outputs", style={'color': '#d63384'}),
        html.Div(debug_components)
    ])


def _display_debug_item(key: str, value: Any) -> html.Div:
    """Display a single debug item (key-value pair).

    Args:
        key: The key/name of the debug item
        value: The debug value to display

    Returns:
        html.Div containing the item visualization
    """
    components = [html.H6(f"{key}:", style={'margin-bottom': '5px'})]

    if isinstance(value, torch.Tensor):
        components.extend(_display_tensor(value))
    elif isinstance(value, np.ndarray):
        components.extend(_display_numpy_array(value))
    elif isinstance(value, dict):
        components.append(_display_dict(value))
    elif isinstance(value, (list, tuple)):
        components.append(_display_list(value))
    else:
        # Display as formatted text
        components.append(
            html.Pre(
                format_value(value),
                style={
                    'background-color': '#f8f9fa',
                    'padding': '8px',
                    'margin': '5px 0',
                    'border-radius': '3px',
                    'font-size': '12px'
                }
            )
        )

    return html.Div(components, style={'margin': '10px 0'})


def _display_tensor(tensor: torch.Tensor) -> List[html.Div]:
    """Display a PyTorch tensor."""
    components = []

    # Show tensor info
    info_text = f"Tensor: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}"
    components.append(
        html.P(info_text, style={'font-size': '12px', 'color': '#666'})
    )

    # Try to display as image if appropriate shape
    if len(tensor.shape) in [2, 3]:
        try:
            # Convert to displayable format
            if tensor.device.type == 'cuda':
                tensor = tensor.cpu()

            if len(tensor.shape) == 3 and tensor.shape[0] in [1, 3]:
                # Image-like tensor (C, H, W)
                fig = create_image_display(tensor, title=f"Tensor {tensor.shape}")
                components.append(
                    html.Div([
                        dcc.Graph(figure=fig)
                    ], style={'width': '300px', 'display': 'inline-block'})
                )
            elif len(tensor.shape) == 2:
                # 2D tensor (could be feature map)
                fig = create_image_display(tensor, title=f"Tensor {tensor.shape}", colorscale="Viridis")
                components.append(
                    html.Div([
                        dcc.Graph(figure=fig)
                    ], style={'width': '300px', 'display': 'inline-block'})
                )
        except Exception as e:
            # Fallback to text display if visualization fails
            components.append(
                html.Pre(
                    f"Visualization failed: {str(e)}\nTensor summary: {tensor}",
                    style={'background-color': '#fff3cd', 'padding': '8px', 'font-size': '12px'}
                )
            )
    else:
        # Display tensor summary for other shapes
        components.append(
            html.Pre(
                str(tensor),
                style={'background-color': '#f8f9fa', 'padding': '8px', 'font-size': '12px'}
            )
        )

    return components


def _display_numpy_array(array: np.ndarray) -> List[html.Div]:
    """Display a NumPy array."""
    components = []

    # Show array info
    info_text = f"Array: shape={array.shape}, dtype={array.dtype}"
    components.append(
        html.P(info_text, style={'font-size': '12px', 'color': '#666'})
    )

    # Try to display as image if appropriate shape
    if len(array.shape) in [2, 3]:
        try:
            # Convert to tensor for visualization
            tensor = torch.from_numpy(array)
            fig = create_image_display(tensor, title=f"Array {array.shape}", colorscale="Viridis")
            components.append(
                html.Div([
                    dcc.Graph(figure=fig)
                ], style={'width': '300px', 'display': 'inline-block'})
            )
        except Exception:
            # Fallback to text display
            components.append(
                html.Pre(
                    str(array),
                    style={'background-color': '#f8f9fa', 'padding': '8px', 'font-size': '12px'}
                )
            )
    else:
        # Display array summary
        components.append(
            html.Pre(
                str(array),
                style={'background-color': '#f8f9fa', 'padding': '8px', 'font-size': '12px'}
            )
        )

    return components


def _display_dict(data: Dict[str, Any]) -> html.Div:
    """Display a dictionary."""
    return html.Pre(
        format_value(data),
        style={
            'background-color': '#f8f9fa',
            'padding': '8px',
            'margin': '5px 0',
            'border-radius': '3px',
            'font-size': '12px'
        }
    )


def _display_list(data: Union[List, tuple]) -> html.Div:
    """Display a list or tuple."""
    return html.Pre(
        format_value(data),
        style={
            'background-color': '#f8f9fa',
            'padding': '8px',
            'margin': '5px 0',
            'border-radius': '3px',
            'font-size': '12px'
        }
    )