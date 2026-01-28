"""Update individual score maps when the epoch slider changes."""

from typing import Dict, List, Optional

import numpy as np
from dash import Dash, Input, Output, State, html
from dash.exceptions import PreventUpdate

from runners.viewers.eval_viewer.backend.initialization import LogDirInfo
from runners.viewers.eval_viewer.callbacks.individual_score_map_helpers import (
    build_individual_score_map_children,
)


def _validate_trigger(epoch: Optional[int], metric_name: Optional[str]) -> None:
    if epoch is None or metric_name is None:
        raise PreventUpdate


def register_individual_score_maps_from_epoch_callback(
    app: Dash,
    metric_names: List[str],
    num_datapoints: int,
    log_dir_infos: Dict[str, LogDirInfo],
    per_metric_color_scales: np.ndarray,
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
        num_datapoints, int
    ), f"num_datapoints must be int, got {type(num_datapoints)}"
    assert num_datapoints > 0, "num_datapoints must be positive"
    assert isinstance(
        log_dir_infos, dict
    ), f"log_dir_infos must be dict, got {type(log_dir_infos)}"
    assert log_dir_infos, "log_dir_infos must be non-empty"
    assert isinstance(
        per_metric_color_scales, np.ndarray
    ), f"per_metric_color_scales must be np.ndarray, got {type(per_metric_color_scales)}"
    assert per_metric_color_scales.shape == (
        len(metric_names),
        2,
    ), "per_metric_color_scales must be shape (len(metric_names), 2)"

    outputs = []
    for i in range(len(log_dir_infos)):
        outputs.append(
            Output(f"individual-button-grid-{i}", "children", allow_duplicate=True)
        )
        outputs.append(
            Output(f"individual-color-bar-{i}", "children", allow_duplicate=True)
        )

    @app.callback(
        outputs,
        Input("epoch-slider", "value"),
        State("metric-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _update_individual_score_maps_from_epoch(
        epoch: Optional[int], metric_name: Optional[str]
    ) -> List[html.Div]:
        _validate_trigger(epoch=epoch, metric_name=metric_name)
        return build_individual_score_map_children(
            epoch=epoch,
            metric_name=metric_name,
            metric_names=metric_names,
            num_datapoints=num_datapoints,
            log_dir_infos=log_dir_infos,
            per_metric_color_scales=per_metric_color_scales,
        )
