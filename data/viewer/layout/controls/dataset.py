"""Utility functions for the dataset viewer."""
from dash import dcc, html


def create_dataset_selector(available_datasets):
    """
    Create a dataset selector dropdown.

    Args:
        available_datasets: Dictionary of available datasets

    Returns:
        html.Div containing the dataset selector
    """
    return html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[{'label': name, 'value': name} for name in sorted(available_datasets.keys())],
            value=None,
            style={'width': '100%'}
        )
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
