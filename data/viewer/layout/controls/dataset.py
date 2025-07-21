"""Utility functions for the dataset viewer."""
from typing import Dict, Any, Union
from dash import dcc, html


def create_dataset_selector(available_datasets: Union[Dict[str, Any], Dict[str, Dict[str, str]]], hierarchical: bool = True) -> html.Div:
    """
    Create a dataset selector dropdown, with optional hierarchical grouping.

    Args:
        available_datasets: Dictionary of available datasets (flat or hierarchical)
        hierarchical: Whether to use hierarchical grouping (default True)

    Returns:
        html.Div containing the dataset selector(s)
    """
    if hierarchical and isinstance(available_datasets, dict) and available_datasets:
        # Check if this is hierarchical data (nested dicts)
        first_value = next(iter(available_datasets.values()))
        if isinstance(first_value, dict):
            return create_hierarchical_selector(available_datasets)
    
    # Fallback to flat dropdown
    return create_flat_selector(available_datasets)


def create_flat_selector(available_datasets: Dict[str, Any]) -> html.Div:
    """Create a traditional flat dataset selector."""
    return html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[{'label': name, 'value': name} for name in sorted(available_datasets.keys())],
            value=None,
            style={'width': '100%'}
        )
    ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'})


def create_hierarchical_selector(hierarchical_datasets: Dict[str, Dict[str, str]]) -> html.Div:
    """Create a hierarchical dataset selector with group and dataset dropdowns."""
    
    # Create type labels mapping
    type_labels = {
        'semseg': 'Semantic Segmentation',
        '2dcd': '2D Change Detection', 
        '3dcd': '3D Change Detection',
        'pcr': 'Point Cloud Registration'
    }
    
    # Create group dropdown options
    group_options = []
    for dataset_type in sorted(hierarchical_datasets.keys()):
        label = type_labels.get(dataset_type, dataset_type.upper())
        group_options.append({'label': f"{label} ({len(hierarchical_datasets[dataset_type])} datasets)", 'value': dataset_type})
    
    return html.Div([
        html.Div([
            html.Label("Select Dataset Group:"),
            dcc.Dropdown(
                id='dataset-group-dropdown',
                options=group_options,
                value=None,
                placeholder="Choose a dataset category...",
                style={'width': '100%'}
            )
        ], style={'margin-bottom': '15px'}),
        
        html.Div([
            html.Label("Select Dataset:"),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=[],
                value=None,
                placeholder="First select a category above...",
                style={'width': '100%'},
                disabled=True
            )
        ])
    ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'})


def create_reload_button():
    """
    Create a button to reload datasets.

    Returns:
        html.Div containing the reload button
    """
    return html.Div([
        html.Button(
            "Reload Datasets",
            id='reload-button',
            style={
                'background-color': '#007bff',
                'color': 'white',
                'border': 'none',
                'padding': '10px 15px',
                'cursor': 'pointer',
                'border-radius': '5px',
                'margin-top': '20px'
            }
        ),
    ], style={'width': '30%', 'display': 'inline-block', 'text-align': 'right'})
