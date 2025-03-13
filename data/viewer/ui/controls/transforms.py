"""UI components for handling dataset transforms."""
from dash import dcc, html


def create_transform_checkboxes(transforms):
    """
    Create checkboxes for each transform in the list.
    
    Args:
        transforms: List of transform configurations
        
    Returns:
        List of html.Div elements containing transform checkboxes
    """
    if not transforms:
        return [html.P("No transforms available")]
    
    transform_checkboxes = []
    for i, transform in enumerate(transforms):
        # Get transform name from class (removing the module path if present)
        transform_class = transform.get('class', 'Unknown')
        if isinstance(transform_class, str):
            class_name = transform_class.split('.')[-1]
        else:
            try:
                class_name = transform_class.__name__
            except AttributeError:
                class_name = str(transform_class)
        
        # Extract transform arguments for display
        transform_args = transform.get('args', {})
        args_str = ", ".join(f"{k}={v}" for k, v in transform_args.items())
        if args_str:
            display_name = f"{class_name}({args_str})"
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


def create_transforms_section(dataset_config=None):
    """
    Create the transforms section with checkboxes.
    
    Args:
        dataset_config: Dataset configuration dictionary
        
    Returns:
        html.Div containing the transforms section
    """
    if not dataset_config:
        return html.Div([
            html.H3("Transforms", style={'margin-top': '0'}),
            html.P("No dataset selected or no transforms available.")
        ])
    
    # Get the transforms configuration
    dataset_cfg = dataset_config.get('train_dataset', {})
    transforms_cfg = dataset_cfg.get('args', {}).get('transforms_cfg')

    if not transforms_cfg or 'args' not in transforms_cfg or 'transforms' not in transforms_cfg['args']:
        return html.Div([
            html.H3("Transforms", style={'margin-top': '0'}),
            html.P("No transforms available for this dataset.")
        ])

    # Create checkboxes for each transform
    transforms = transforms_cfg['args']['transforms']
    transform_checkboxes = create_transform_checkboxes(transforms)

    return html.Div([
        html.H3("Transforms", style={'margin-top': '0'}),
        html.Div(transform_checkboxes, style={'max-height': '200px', 'overflow-y': 'auto'})
    ])
