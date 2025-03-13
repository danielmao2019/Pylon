"""UI components for handling dataset transforms."""
from typing import Dict, List, Optional, Union, Tuple, Any
from dash import dcc, html


def create_transform_checkboxes(transforms: List[Tuple]) -> List[html.Div]:
    """
    Create checkboxes for each transform in the list.
    
    Args:
        transforms: List of transform configurations as tuples of (transform, paths)
        where transform can be either a dict config or an actual transform instance
        
    Returns:
        List of html.Div elements containing transform checkboxes
    """
    if not transforms:
        return [html.P("No transforms available")]
    
    transform_checkboxes: List[html.Div] = []
    for i, (transform, _) in enumerate(transforms):
        class_name = transform.__class__.__name__
        # Get the transform arguments from the instance
        transform_args = {
            k: v for k, v in transform.__dict__.items()
            if not k.startswith('_')  # Skip private attributes
        }
        
        # Create display name
        args_str: str = ", ".join(f"{k}={v}" for k, v in transform_args.items())
        if args_str:
            display_name: str = f"{class_name}({args_str})"
        else:
            display_name = class_name
        
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
