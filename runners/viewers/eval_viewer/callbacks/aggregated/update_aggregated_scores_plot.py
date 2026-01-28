"""Update the aggregated scores plot when the metric changes."""

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc
from dash.exceptions import PreventUpdate

from runners.viewers.eval_viewer.backend.initialization import LogDirInfo


def create_aggregated_scores_plot(
    epoch_scores: List[np.ndarray], log_dirs: List[str], metric_name: str
) -> go.Figure:
    # Input validations
    assert isinstance(
        epoch_scores, list
    ), f"epoch_scores must be list, got {type(epoch_scores)}"
    assert epoch_scores, "epoch_scores must be non-empty"
    assert all(
        isinstance(scores, np.ndarray) for scores in epoch_scores
    ), "epoch_scores must contain numpy arrays"
    assert isinstance(log_dirs, list), f"log_dirs must be list, got {type(log_dirs)}"
    assert log_dirs, "log_dirs must be non-empty"
    assert len(log_dirs) == len(
        epoch_scores
    ), "log_dirs length must match epoch_scores length"
    assert all(
        isinstance(log_dir, str) for log_dir in log_dirs
    ), "log_dirs must contain strings"
    assert isinstance(
        metric_name, str
    ), f"metric_name must be str, got {type(metric_name)}"

    fig = go.Figure()
    for scores, log_dir in zip(epoch_scores, log_dirs, strict=True):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(scores))),
                y=scores,
                name=log_dir.split("/")[-1],
                mode="lines+markers",
            )
        )

    fig.update_layout(
        title=f"Aggregated {metric_name} Over Time",
        xaxis_title="Epoch",
        yaxis_title="Score",
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def _validate_trigger(metric_name: Optional[str]) -> None:
    if metric_name is None:
        raise PreventUpdate


def _validate_inputs(
    metric_name: str, metric_names: List[str], log_dir_infos: Dict[str, LogDirInfo]
) -> None:
    # Input validations
    assert isinstance(
        metric_name, str
    ), f"metric_name must be str, got {type(metric_name)}"
    assert isinstance(
        metric_names, list
    ), f"metric_names must be list, got {type(metric_names)}"
    assert metric_names, "metric_names must be non-empty"
    assert all(
        isinstance(name, str) for name in metric_names
    ), "metric_names must contain strings"
    assert (
        metric_name in metric_names
    ), f"metric_name {metric_name} not found in metrics"
    assert isinstance(
        log_dir_infos, dict
    ), f"log_dir_infos must be dict, got {type(log_dir_infos)}"
    assert log_dir_infos, "log_dir_infos must be non-empty"


def register_aggregated_scores_plot_callback(
    app: Dash, metric_names: List[str], log_dir_infos: Dict[str, LogDirInfo]
) -> None:
    # Input validations
    assert isinstance(app, Dash), f"app must be Dash, got {type(app)}"
    assert isinstance(
        metric_names, list
    ), f"metric_names must be list, got {type(metric_names)}"
    assert metric_names, "metric_names must be non-empty"
    assert all(
        isinstance(name, str) for name in metric_names
    ), "metric_names must contain strings"
    assert isinstance(
        log_dir_infos, dict
    ), f"log_dir_infos must be dict, got {type(log_dir_infos)}"
    assert log_dir_infos, "log_dir_infos must be non-empty"

    @app.callback(
        Output("aggregated-scores-plot", "children"),
        Input("metric-dropdown", "value"),
    )
    def _update_aggregated_scores_plot(metric_name: Optional[str]) -> dcc.Graph:
        _validate_trigger(metric_name=metric_name)
        _validate_inputs(
            metric_name=metric_name,
            metric_names=metric_names,
            log_dir_infos=log_dir_infos,
        )

        metric_idx = metric_names.index(metric_name)
        epoch_scores: List[np.ndarray] = []
        for info in log_dir_infos.values():
            if info.runner_type == "trainer":
                epoch_scores.append(info.aggregated_scores[:, metric_idx])
            elif info.runner_type == "evaluator":
                epoch_scores.append(np.array([info.aggregated_scores[metric_idx]]))
            else:
                raise ValueError(f"Unknown runner type: {info.runner_type}")

        fig = create_aggregated_scores_plot(
            epoch_scores=epoch_scores,
            log_dirs=list(log_dir_infos.keys()),
            metric_name=metric_name,
        )
        return dcc.Graph(figure=fig)
