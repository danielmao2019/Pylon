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
        return [html.P("No transforms available")]

    transform_checkboxes: List[html.Div] = []
    for transform in transforms:
        transform_checkboxes.append(
            html.Div([
                dcc.Checklist(
                    id={'type': 'transform-checkbox', 'index': transform['index']},
                    options=[{'label': transform['name'], 'value': transform['index']}],
                    value=[],
                    style={'display': 'inline-block', 'margin-right': '10px'}
                ),
                html.Div(
                    transform['description'],
                    style={'font-size': 'small', 'color': 'gray', 'margin-left': '25px'}
                )
            ], style={'margin': '5px 0'})
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
        html.Div(create_transform_checkboxes(transforms), style={'max-height': '200px', 'overflow-y': 'auto'})
    ])
