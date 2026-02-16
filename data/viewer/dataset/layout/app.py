"""App layout components for the dataset viewer."""

from dash import Dash, dcc, html

from data.viewer.dataset.context import get_viewer_context
from data.viewer.dataset.layout.controls.controls_3d import create_3d_controls
from data.viewer.dataset.layout.controls.dataset import (
    create_dataset_selector,
    create_reload_button,
)
from data.viewer.dataset.layout.controls.navigation import create_navigation_controls


def build_layout(app: Dash) -> None:
    # Input validations
    assert isinstance(app, Dash), f"app must be Dash, got {type(app)}"

    available_datasets = get_viewer_context().available_datasets
    layout = html.Div(
        [
            dcc.Store(id="dataset-info", data={}),
            dcc.Store(id="transforms-store", data=None),
            dcc.Store(id="3d-settings-store", data={}),
            dcc.Store(id="camera-state", data=None),
            dcc.Store(id="backend-sync-3d-settings", data={}),
            dcc.Store(id="backend-sync-dataset", data={}),
            dcc.Store(id="backend-sync-navigation", data={}),
            html.Div(
                [
                    html.H1(
                        "Dataset Viewer",
                        style={"text-align": "center", "margin-bottom": "20px"},
                    ),
                    html.Div(
                        [
                            create_dataset_selector(available_datasets),
                            create_reload_button(),
                        ],
                        style={"display": "flex", "align-items": "flex-end"},
                    ),
                    create_navigation_controls(),
                ],
                style={
                    "padding": "20px",
                    "background-color": "#f8f9fa",
                    "border-radius": "5px",
                    "margin-bottom": "20px",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                id="dataset-info-display",
                                style={"margin-bottom": "20px"},
                            ),
                            html.Div(
                                id="transforms-section", style={"margin-bottom": "20px"}
                            ),
                            create_3d_controls(visible=False),
                        ],
                        style={
                            "width": "25%",
                            "padding": "20px",
                            "background-color": "#f8f9fa",
                            "border-radius": "5px",
                        },
                    ),
                    html.Div(
                        [html.Div(id="datapoint-display", style={"padding": "10px"})],
                        style={
                            "width": "75%",
                            "padding": "20px",
                            "background-color": "#ffffff",
                            "border-radius": "5px",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "20px"},
            ),
        ]
    )
    app.layout = layout
