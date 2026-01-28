"""Layout components for the training losses viewer."""

from dash import Dash, dcc, html

from runners.viewers.train_viewer.layout.styles import (
    APP_CONTAINER_STYLE,
    CONTROL_BUTTON_STYLE,
    CONTROL_PANEL_STYLE,
    CONTROL_TITLE_STYLE,
    LABEL_STYLE,
    PLOTS_PANEL_STYLE,
    SMOOTHING_INFO_STYLE,
    TITLE_STYLE,
)


def build_layout(app: Dash) -> None:
    # Input validations
    assert isinstance(app, Dash), f"app must be Dash, got {type(app)}"

    layout = html.Div(
        [
            html.H1("Training Losses Viewer", style=TITLE_STYLE),
            _build_content_layout(),
        ]
    )
    app.layout = layout


def _build_content_layout() -> html.Div:
    return html.Div(
        [
            _build_control_panel(),
            _build_plots_panel(),
        ],
        style=APP_CONTAINER_STYLE,
    )


def _build_control_panel() -> html.Div:
    return html.Div(
        [
            html.H3("Controls", style=CONTROL_TITLE_STYLE),
            html.Button(
                "Refresh",
                id="refresh-button",
                n_clicks=0,
                style=CONTROL_BUTTON_STYLE,
            ),
            html.Hr(),
            html.Label("Loss Smoothing:", style=LABEL_STYLE),
            dcc.Slider(
                id="smoothing-slider",
                min=1,
                max=50,
                step=1,
                value=1,
                marks={1: "1", 10: "10", 25: "25", 50: "50"},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Div(
                id="smoothing-info",
                style=SMOOTHING_INFO_STYLE,
            ),
        ],
        style=CONTROL_PANEL_STYLE,
    )


def _build_plots_panel() -> html.Div:
    return html.Div(
        [
            html.Div(id="plots-container"),
        ],
        style=PLOTS_PANEL_STYLE,
    )
