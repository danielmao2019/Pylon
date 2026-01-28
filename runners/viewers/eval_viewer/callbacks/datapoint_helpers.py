"""Helpers for datapoint viewer callbacks."""

from typing import Any, Dict, List

import dash
from dash import html
from dash.exceptions import PreventUpdate

from data.viewer.backend.backend import DatasetType
from runners.viewers.eval_viewer.backend.initialization import (
    LogDirInfo,
    load_debug_outputs,
)
from utils.builders.builder import build_from_config


def validate_datapoint_trigger(
    overlaid_clicks: List[int],
    individual_clicks: List[int],
    epoch: int,
    metric_name: str,
) -> Dict[str, Any]:
    if (
        not any(overlaid_clicks)
        and not any(individual_clicks)
        or epoch is None
        or metric_name is None
    ):
        raise PreventUpdate

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered_id
    if not isinstance(triggered_id, dict) or "index" not in triggered_id:
        raise PreventUpdate

    return triggered_id


def validate_datapoint_inputs(
    overlaid_clicks: List[int],
    individual_clicks: List[int],
    epoch: int,
    metric_name: str,
    dataset_type: DatasetType,
    log_dir_infos: Dict[str, LogDirInfo],
) -> None:
    # Input validations
    assert isinstance(
        overlaid_clicks, list
    ), f"overlaid_clicks must be list, got {type(overlaid_clicks)}"
    assert isinstance(
        individual_clicks, list
    ), f"individual_clicks must be list, got {type(individual_clicks)}"
    assert isinstance(epoch, int), f"epoch must be int, got {type(epoch)}"
    assert epoch >= 0, "epoch must be non-negative"
    assert isinstance(
        metric_name, str
    ), f"metric_name must be str, got {type(metric_name)}"
    assert isinstance(
        dataset_type, str
    ), f"dataset_type must be str, got {type(dataset_type)}"
    assert isinstance(
        log_dir_infos, dict
    ), f"log_dir_infos must be dict, got {type(log_dir_infos)}"
    assert log_dir_infos, "log_dir_infos must be non-empty"


def build_datapoint_display(
    dataset: Any,
    dataset_type: DatasetType,
    log_dir_infos: Dict[str, LogDirInfo],
    epoch: int,
    triggered_id: Dict[str, Any],
) -> html.Div:
    # Input validations
    assert dataset is not None, "dataset must not be None"
    assert isinstance(
        dataset_type, str
    ), f"dataset_type must be str, got {type(dataset_type)}"
    assert isinstance(
        log_dir_infos, dict
    ), f"log_dir_infos must be dict, got {type(log_dir_infos)}"
    assert log_dir_infos, "log_dir_infos must be non-empty"
    assert isinstance(epoch, int), f"epoch must be int, got {type(epoch)}"
    assert epoch >= 0, "epoch must be non-negative"
    assert isinstance(
        triggered_id, dict
    ), f"triggered_id must be dict, got {type(triggered_id)}"
    assert "index" in triggered_id, "triggered_id must contain index"

    index_parts = triggered_id["index"].split("-")

    if triggered_id["type"] == "overlaid-grid-button":
        run_idx = None
        datapoint_idx = int(index_parts[0])
        current_dataset = dataset
        datapoint = current_dataset[datapoint_idx]
    else:
        run_idx = int(index_parts[0])
        datapoint_idx = int(index_parts[1])
        run_info = list(log_dir_infos.values())[run_idx]
        current_dataset = build_from_config(run_info.dataset_cfg)
        dataloader = build_from_config(run_info.dataloader_cfg, dataset=current_dataset)
        collate_fn = dataloader.collate_fn
        datapoint = current_dataset[datapoint_idx]
        datapoint = collate_fn([datapoint])

        log_dir = list(log_dir_infos.keys())[run_idx]
        if run_info.runner_type == "trainer":
            epoch_dir = f"{log_dir}/epoch_{epoch}"
            debug_outputs = load_debug_outputs(epoch_dir)
        else:
            debug_outputs = load_debug_outputs(log_dir)

        if debug_outputs and datapoint_idx in debug_outputs:
            datapoint["debug"] = debug_outputs[datapoint_idx]

    assert hasattr(
        current_dataset, "display_datapoint"
    ), f"Dataset {type(current_dataset).__name__} must have display_datapoint method"
    display = current_dataset.display_datapoint(datapoint)

    return html.Div(
        [
            html.Div(
                [
                    html.H4(f"Datapoint {datapoint_idx}"),
                    html.P(f"Type: {dataset_type}"),
                    html.P(
                        f"Source: {'Individual Run' if run_idx is not None else 'Overlaid View'}"
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(display),
        ]
    )
