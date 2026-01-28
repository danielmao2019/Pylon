"""Layout components for the evaluation viewer."""

from typing import List, Optional

import numpy as np
from dash import Dash, dcc, html

from runners.viewers.eval_viewer.backend.visualization import get_color_for_score
from runners.viewers.eval_viewer.layout.styles import (
    AGGREGATED_PLOT_CONTAINER_STYLE,
    AGGREGATED_PLOT_STYLE,
    APP_TITLE_STYLE,
    COLOR_BAR_CONTAINER_STYLE,
    COLOR_BAR_GRADIENT_STYLE,
    COLOR_BAR_LABELS_STYLE,
    COLOR_BAR_ROW_STYLE,
    CONTROL_SECTION_RIGHT_STYLE,
    CONTROL_SECTION_STYLE,
    CONTROLS_CONTAINER_STYLE,
    DATAPOINT_DISPLAY_STYLE,
    DATAPOINT_SECTION_STYLE,
    GRID_BUTTON_NAN_STYLE,
    GRID_BUTTON_VALID_STYLE,
    GRID_CONTAINER_BASE_STYLE,
    GRID_PADDING_STYLE,
    INDIVIDUAL_BUTTON_GRID_STYLE,
    INDIVIDUAL_COLOR_BAR_STYLE,
    INDIVIDUAL_RUN_CONTAINER_STYLE,
    INDIVIDUAL_RUN_ROW_STYLE,
    INDIVIDUAL_RUN_TITLE_STYLE,
    INDIVIDUAL_SCORE_MAPS_CONTAINER_STYLE,
    LEFT_COLUMN_STYLE,
    MAIN_CONTENT_STYLE,
    OVERLAID_GRID_STYLE,
    OVERLAID_ROW_STYLE,
    OVERLAID_SECTION_STYLE,
    RIGHT_COLUMN_STYLE,
    SECTION_TITLE_STYLE,
)


def build_layout(
    app: Dash, max_epochs: int, metric_names: List[str], run_names: List[str]
) -> None:
    # Input validations
    assert isinstance(app, Dash), f"app must be Dash, got {type(app)}"
    assert isinstance(
        max_epochs, int
    ), f"max_epochs must be int, got {type(max_epochs)}"
    assert max_epochs > 0, "max_epochs must be positive"
    assert isinstance(
        metric_names, list
    ), f"metric_names must be list, got {type(metric_names)}"
    assert metric_names, "metric_names must be non-empty"
    assert all(
        isinstance(metric, str) for metric in metric_names
    ), "metric_names must contain strings"
    assert all(metric_names), "metric_names must contain non-empty strings"
    assert isinstance(run_names, list), f"run_names must be list, got {type(run_names)}"
    assert run_names, "run_names must be non-empty"
    assert all(
        isinstance(name, str) for name in run_names
    ), "run_names must contain strings"
    assert all(run_names), "run_names must contain non-empty strings"

    layout = html.Div(
        [
            html.H1("Evaluation Viewer", style=APP_TITLE_STYLE),
            _build_main_content(
                max_epoch=max_epochs, metric_names=metric_names, run_names=run_names
            ),
        ]
    )
    app.layout = layout


def _build_main_content(
    max_epoch: int, metric_names: List[str], run_names: List[str]
) -> html.Div:
    return html.Div(
        [
            _build_left_column(
                max_epoch=max_epoch, metric_names=metric_names, run_names=run_names
            ),
            _build_right_column(),
        ],
        style=MAIN_CONTENT_STYLE,
    )


def _build_left_column(
    max_epoch: int, metric_names: List[str], run_names: List[str]
) -> html.Div:
    return html.Div(
        [
            _build_controls(max_epoch=max_epoch, metric_names=metric_names),
            _build_aggregated_scores_plot(),
            _build_score_maps_grid(run_names=run_names),
        ],
        style=LEFT_COLUMN_STYLE,
    )


def _build_controls(max_epoch: int, metric_names: List[str]) -> html.Div:
    # Input validations
    assert isinstance(max_epoch, int), f"max_epoch must be int, got {type(max_epoch)}"
    assert max_epoch > 0, "max_epoch must be positive"
    assert isinstance(
        metric_names, list
    ), f"metric_names must be list, got {type(metric_names)}"
    assert metric_names, "metric_names must be non-empty"
    assert all(
        isinstance(metric, str) for metric in metric_names
    ), "metric_names must contain strings"
    assert all(metric_names), "metric_names must contain non-empty strings"

    return html.Div(
        [
            html.Div(
                [
                    html.Label("Epoch:"),
                    dcc.Slider(
                        id="epoch-slider",
                        min=0,
                        max=max_epoch - 1,
                        step=1,
                        value=0,
                        marks={i: str(i) for i in range(max_epoch)},
                        updatemode="drag",
                    ),
                ],
                style=CONTROL_SECTION_STYLE,
            ),
            html.Div(
                [
                    html.Label("Metric:"),
                    dcc.Dropdown(
                        id="metric-dropdown",
                        options=[
                            {"label": metric, "value": metric}
                            for metric in sorted(metric_names)
                        ],
                        value=metric_names[0],
                    ),
                ],
                style=CONTROL_SECTION_STYLE,
            ),
            html.Div(
                [
                    html.Label("Failure Percentile:"),
                    dcc.Slider(
                        id="percentile-slider",
                        min=5,
                        max=95,
                        step=5,
                        value=25,
                        marks={i: f"{i}%" for i in range(5, 100, 10)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode="drag",
                    ),
                ],
                style=CONTROL_SECTION_RIGHT_STYLE,
            ),
        ],
        style=CONTROLS_CONTAINER_STYLE,
    )


def _build_aggregated_scores_plot() -> html.Div:
    return html.Div(
        [
            html.H2("Aggregated Scores Over Time", style=SECTION_TITLE_STYLE),
            html.Div(id="aggregated-scores-plot", style=AGGREGATED_PLOT_STYLE),
        ],
        style=AGGREGATED_PLOT_CONTAINER_STYLE,
    )


def _build_score_maps_grid(run_names: List[str]) -> html.Div:
    return html.Div(
        [
            _build_overlaid_score_map_layout(),
            _build_individual_score_maps_layout(run_names=run_names),
        ]
    )


def _build_overlaid_score_map_layout() -> html.Div:
    return html.Div(
        [
            html.H2("Common Failure Cases", style=SECTION_TITLE_STYLE),
            html.Div(
                [
                    html.Div(id="overlaid-button-grid", style=OVERLAID_GRID_STYLE),
                    html.Div(id="overlaid-color-bar", style=INDIVIDUAL_COLOR_BAR_STYLE),
                ],
                style=OVERLAID_ROW_STYLE,
            ),
        ],
        style=OVERLAID_SECTION_STYLE,
    )


def _build_individual_score_maps_layout(run_names: List[str]) -> html.Div:
    # Input validations
    assert isinstance(run_names, list), f"run_names must be list, got {type(run_names)}"
    assert run_names, "run_names must be non-empty"
    assert all(
        isinstance(name, str) for name in run_names
    ), "run_names must contain strings"

    return html.Div(
        [
            html.H2("Individual Score Maps", style=SECTION_TITLE_STYLE),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(run_name, style=INDIVIDUAL_RUN_TITLE_STYLE),
                            html.Div(
                                [
                                    html.Div(
                                        id=f"individual-button-grid-{i}",
                                        style=INDIVIDUAL_BUTTON_GRID_STYLE,
                                    ),
                                    html.Div(
                                        id=f"individual-color-bar-{i}",
                                        style=INDIVIDUAL_COLOR_BAR_STYLE,
                                    ),
                                ],
                                style=INDIVIDUAL_RUN_ROW_STYLE,
                            ),
                        ],
                        style=INDIVIDUAL_RUN_CONTAINER_STYLE,
                    )
                    for i, run_name in enumerate(run_names)
                ],
                style=INDIVIDUAL_SCORE_MAPS_CONTAINER_STYLE,
            ),
        ],
        style=OVERLAID_SECTION_STYLE,
    )


def _build_right_column() -> html.Div:
    return html.Div(
        [
            _build_datapoint_display_section(),
        ],
        style=RIGHT_COLUMN_STYLE,
    )


def _build_datapoint_display_section() -> html.Div:
    return html.Div(
        [
            html.H3("Selected Datapoint", style=SECTION_TITLE_STYLE),
            html.Div(id="datapoint-display", style=DATAPOINT_DISPLAY_STYLE),
        ],
        style=DATAPOINT_SECTION_STYLE,
    )


def create_color_bar(min_score: float, max_score: float) -> html.Div:
    # Input validations
    assert isinstance(
        min_score, (int, float)
    ), f"min_score must be numeric, got {type(min_score)}"
    assert isinstance(
        max_score, (int, float)
    ), f"max_score must be numeric, got {type(max_score)}"

    return html.Div(
        [
            html.Div(
                [
                    html.Div(style=COLOR_BAR_GRADIENT_STYLE),
                    html.Div(
                        [
                            html.Div(f"{max_score:.2f}", style={"marginBottom": "5px"}),
                            html.Div(f"{min_score:.2f}"),
                        ],
                        style=COLOR_BAR_LABELS_STYLE,
                    ),
                ],
                style=COLOR_BAR_ROW_STYLE,
            )
        ],
        style=COLOR_BAR_CONTAINER_STYLE,
    )


def create_button_grid(
    num_datapoints: int,
    score_map: np.ndarray,
    button_type: str,
    run_idx: Optional[int] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
) -> html.Div:
    # Input validations
    assert isinstance(
        num_datapoints, int
    ), f"num_datapoints must be int, got {type(num_datapoints)}"
    assert num_datapoints > 0, "num_datapoints must be positive"
    assert isinstance(
        score_map, np.ndarray
    ), f"score_map must be np.ndarray, got {type(score_map)}"
    assert score_map.ndim == 2, f"score_map must be 2D, got shape {score_map.shape}"
    assert isinstance(
        button_type, str
    ), f"button_type must be str, got {type(button_type)}"
    assert run_idx is None or isinstance(run_idx, int), "run_idx must be int or None"
    assert min_score is None or isinstance(
        min_score, (int, float)
    ), "min_score must be numeric or None"
    assert max_score is None or isinstance(
        max_score, (int, float)
    ), "max_score must be numeric or None"

    side_length = score_map.shape[0]

    if min_score is None:
        min_score = np.nanmin(score_map)
    if max_score is None:
        max_score = np.nanmax(score_map)

    buttons = []
    for row in range(side_length):
        for col in range(side_length):
            idx = row * side_length + col
            if idx >= num_datapoints:
                buttons.append(html.Div(style=GRID_PADDING_STYLE))
                continue

            value = score_map[row, col]
            button_id = {
                "type": button_type,
                "index": f"{run_idx}-{idx}" if run_idx is not None else str(idx),
            }

            if np.isnan(value):
                button_style = GRID_BUTTON_NAN_STYLE
                button = html.Button("", id=button_id, style=button_style)
            else:
                color = get_color_for_score(
                    score=value, min_score=min_score, max_score=max_score
                )
                button_style = {**GRID_BUTTON_VALID_STYLE, "backgroundColor": color}
                button = html.Button("", id=button_id, style=button_style)

            buttons.append(button)

    grid_style = {
        **GRID_CONTAINER_BASE_STYLE,
        "gridTemplateColumns": f"repeat({side_length}, 20px)",
    }
    return html.Div(buttons, style=grid_style)
