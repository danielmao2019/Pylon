"""UI components for handling dataset transforms."""
from typing import Dict, List, Optional, Union, Any
from dash import dcc, html


def create_transform_checkboxes(transforms: List[Dict[str, Any]]) -> List[html.Div]:
    """
    Create checkboxes for each transform in the list.

    Args:
        transforms: List of transform info dictionaries

    Returns:
        List of html.Div elements containing transform checkboxes
    """
    if not transforms:
        return [html.P("No transforms available", style={'color': '#666', 'font-style': 'italic'})]

    transform_checkboxes: List[html.Div] = []
    for transform in transforms:
        # Get the transform string representation
        transform_string = transform.get('string', transform.get('name', 'Unknown'))
        transform_name = transform.get('name', 'Unknown')

        # Create input/output info
        input_names = transform.get('input_names', [])
        output_names = transform.get('output_names', [])

        input_str = ", ".join([f"{inp[0]}.{inp[1]}" for inp in input_names])
        output_str = ", ".join([f"{out[0]}.{out[1]}" for out in output_names])

        # Create tooltip text
        tooltip_text = f"Inputs: {input_str}\nOutputs: {output_str}"

        transform_checkboxes.append(
            html.Div([
                dcc.Checklist(
                    id={'type': 'transform-checkbox', 'index': transform['index']},
                    options=[{'label': '', 'value': transform['index']}],  # Empty label, we'll use custom styling
                    value=[transform['index']],  # Pre-select all transforms by default
                    style={'display': 'inline-block', 'margin-right': '8px'}
                ),
                html.Div([
                    html.Span(
                        transform_string,
                        style={
                            'font-weight': 'bold',
                            'color': '#2c3e50',
                            'font-family': 'monospace'  # Use monospace for code-like appearance
                        }
                    ),
                    html.Br(),
                    html.Span(
                        f"📥 {input_str}",
                        style={
                            'font-size': '0.8em',
                            'color': '#27ae60',
                            'display': 'block',
                            'margin-top': '2px'
                        }
                    ),
                    html.Span(
                        f"📤 {output_str}",
                        style={
                            'font-size': '0.8em',
                            'color': '#e74c3c',
                            'display': 'block',
                            'margin-top': '1px'
                        }
                    ) if output_str != input_str else None
                ], style={'display': 'inline-block', 'vertical-align': 'top'})
            ], style={
                'margin': '8px 0',
                'padding': '8px',
                'border': '1px solid #ecf0f1',
                'border-radius': '4px',
                'background-color': '#f8f9fa',
                'display': 'flex',
                'align-items': 'flex-start'
            }, title=tooltip_text)
        )

    return transform_checkboxes


def create_transforms_section(transforms: Optional[List[Dict[str, Any]]] = None) -> html.Div:
    """
    Create the transforms section with checkboxes.

    Args:
        transforms: Optional list of transform info dictionaries. If None, shows empty state.

    Returns:
        html.Div containing the transforms section
    """
    transforms = transforms or []
    return html.Div([
        html.H3("Transforms", style={'margin-top': '0'}),
        html.Div(create_transform_checkboxes(transforms)),
    ])
