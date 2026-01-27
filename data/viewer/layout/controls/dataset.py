"""Utility functions for the dataset viewer."""

from typing import Dict

from dash import dcc, html

from data.viewer.callbacks.dataset_group_reload import TYPE_LABELS


def create_dataset_selector(
    hierarchical_datasets: Dict[str, Dict[str, str]],
) -> html.Div:
    """Create a hierarchical dataset selector with group and dataset dropdowns."""
    # Input validations
    assert isinstance(
        hierarchical_datasets, dict
    ), f"hierarchical_datasets must be dict, got {type(hierarchical_datasets)}"

    group_options = []
    for dataset_type in sorted(hierarchical_datasets.keys()):
        assert dataset_type in TYPE_LABELS, f"Unknown dataset type: {dataset_type}"
        label = TYPE_LABELS[dataset_type]
        group_options.append(
            {
                'label': f"{label} ({len(hierarchical_datasets[dataset_type])} datasets)",
                'value': dataset_type,
            }
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Label("Select Dataset Group:"),
                    dcc.Dropdown(
                        id='dataset-group-dropdown',
                        options=group_options,
                        value=None,
                        placeholder="Choose a dataset category...",
                        style={'width': '100%'},
                    ),
                ],
                style={'margin-bottom': '15px'},
            ),
            html.Div(
                [
                    html.Label("Select Dataset:"),
                    dcc.Dropdown(
                        id='dataset-dropdown',
                        options=[],
                        value=None,
                        placeholder="First select a category above...",
                        style={'width': '100%'},
                        disabled=True,
                    ),
                ]
            ),
        ],
        style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'},
    )


def create_reload_button() -> html.Div:
    """Create a button to reload datasets."""
    return html.Div(
        [
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
                    'margin-top': '20px',
                },
            ),
        ],
        style={'width': '30%', 'display': 'inline-block', 'text-align': 'right'},
    )
