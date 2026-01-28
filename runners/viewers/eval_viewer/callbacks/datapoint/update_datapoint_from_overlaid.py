"""Update the datapoint display from overlaid grid clicks."""

from typing import Any, Dict, List

import dash
from dash import Dash, Input, Output, State, html

from data.viewer.backend.backend import DatasetType
from runners.viewers.eval_viewer.backend.initialization import LogDirInfo
from runners.viewers.eval_viewer.callbacks.datapoint_helpers import (
    build_datapoint_display,
    validate_datapoint_inputs,
    validate_datapoint_trigger,
)


def register_datapoint_from_overlaid_callback(
    app: Dash,
    dataset: Any,
    dataset_type: DatasetType,
    log_dir_infos: Dict[str, LogDirInfo],
) -> None:
    # Input validations
    assert isinstance(app, Dash), f"app must be Dash, got {type(app)}"
    assert dataset is not None, "dataset must not be None"
    assert isinstance(
        dataset_type, str
    ), f"dataset_type must be str, got {type(dataset_type)}"
    assert isinstance(
        log_dir_infos, dict
    ), f"log_dir_infos must be dict, got {type(log_dir_infos)}"
    assert log_dir_infos, "log_dir_infos must be non-empty"

    @app.callback(
        Output("datapoint-display", "children", allow_duplicate=True),
        Input({"type": "overlaid-grid-button", "index": dash.ALL}, "n_clicks"),
        State({"type": "individual-grid-button", "index": dash.ALL}, "n_clicks"),
        State("epoch-slider", "value"),
        State("metric-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _update_datapoint_from_overlaid(
        overlaid_clicks: List[int],
        individual_clicks: List[int],
        epoch: int,
        metric_name: str,
    ) -> html.Div:
        triggered_id = validate_datapoint_trigger(
            overlaid_clicks=overlaid_clicks,
            individual_clicks=individual_clicks,
            epoch=epoch,
            metric_name=metric_name,
        )
        validate_datapoint_inputs(
            overlaid_clicks=overlaid_clicks,
            individual_clicks=individual_clicks,
            epoch=epoch,
            metric_name=metric_name,
            dataset_type=dataset_type,
            log_dir_infos=log_dir_infos,
        )
        return build_datapoint_display(
            dataset=dataset,
            dataset_type=dataset_type,
            log_dir_infos=log_dir_infos,
            epoch=epoch,
            triggered_id=triggered_id,
        )
