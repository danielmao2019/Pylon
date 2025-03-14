"""UI components for handling dataset transforms."""
from typing import Dict, List, Optional, Union, Tuple, Any
from dash import dcc, html


def create_transform_checkboxes(transforms: List[int]) -> List[html.Div]:
    """
    Create checkboxes for each transform in the list.

    Args:
        transforms: List of transform indices

    Returns:
        List of html.Div elements containing transform checkboxes
    """
    if not transforms:
        return [html.P("No transforms available")]

    # Import registry here to avoid circular import
    from data.viewer.callbacks.registry import registry

    transform_checkboxes: List[html.Div] = []
    for i in transforms:
        # Get transform function from registry
        transform_name = f"transform_{i}"
        transform_func = registry.viewer.dataset_manager._transform_functions.get(transform_name)

        # Create display name from transform function
        if transform_func:
            display_name = transform_func.__class__.__name__
        else:
            display_name = f"Transform {i}"

        # Create the checkbox
        transform_checkboxes.append(
            html.Div([
                dcc.Checklist(
                    id={'type': 'transform-checkbox', 'index': i},
                    options=[{'label': display_name, 'value': i}],
                    value=[],
                    style={'display': 'inline-block', 'margin-right': '10px'}
                )
            ], style={'margin': '5px 0'})
        )

    return transform_checkboxes


def create_transforms_section(transforms_or_config: Optional[Union[Dict, List[Tuple]]] = None) -> html.Div:
    """
    Create the transforms section with checkboxes.

    Args:
        transforms_or_config: Either a dataset configuration dictionary or a list of transforms

    Returns:
        html.Div containing the transforms section
    """
    transforms = []

    if isinstance(transforms_or_config, dict):
        # Handle dataset config case
        dataset_cfg: Dict = transforms_or_config.get('train_dataset', {})
        transforms_cfg: Optional[Dict] = dataset_cfg.get('args', {}).get('transforms_cfg')

        if transforms_cfg and 'args' in transforms_cfg and 'transforms' in transforms_cfg['args']:
            transforms = transforms_cfg['args']['transforms']
    elif isinstance(transforms_or_config, list):
        # Handle direct transforms list case
        transforms = transforms_or_config

    return html.Div([
        html.H3("Transforms", style={'margin-top': '0'}),
        html.Div(create_transform_checkboxes(transforms), style={'max-height': '200px', 'overflow-y': 'auto'})
    ])
