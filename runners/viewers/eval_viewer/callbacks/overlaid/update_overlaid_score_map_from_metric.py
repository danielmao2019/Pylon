"""Update the overlaid score map when the metric dropdown changes."""

from typing import Dict, List, Optional, Tuple

from dash import Dash, Input, Output, State, html

from runners.viewers.eval_viewer.backend.initialization import LogDirInfo
from runners.viewers.eval_viewer.callbacks.overlaid_score_map_helpers import (
    build_overlaid_score_map_children,
    resolve_percentile,
    validate_overlaid_trigger,
)


def register_overlaid_score_map_from_metric_callback(
    app: Dash,
    metric_names: List[str],
    num_datapoints: int,
    log_dir_infos: Dict[str, LogDirInfo],
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

    @app.callback(
        Output("overlaid-button-grid", "children", allow_duplicate=True),
        Output("overlaid-color-bar", "children", allow_duplicate=True),
        Input("metric-dropdown", "value"),
        State("epoch-slider", "value"),
        State("percentile-slider", "drag_value"),
        State("percentile-slider", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def _update_overlaid_score_map_from_metric(
        metric_name: Optional[str],
        epoch: Optional[int],
        percentile_drag: Optional[float],
        percentile_value: Optional[float],
    ) -> Tuple[html.Div, html.Div]:
        validate_overlaid_trigger(
            epoch=epoch,
            metric_name=metric_name,
            percentile_drag=percentile_drag,
            percentile_value=percentile_value,
        )
        percentile = resolve_percentile(
            percentile_drag=percentile_drag, percentile_value=percentile_value
        )
        return build_overlaid_score_map_children(
            epoch=epoch,
            metric_name=metric_name,
            percentile=percentile,
            metric_names=metric_names,
            num_datapoints=num_datapoints,
            log_dir_infos=log_dir_infos,
        )
