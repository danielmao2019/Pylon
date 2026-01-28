"""Helpers for the overlaid score map callbacks."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from dash import html
from dash.exceptions import PreventUpdate

from runners.viewers.eval_viewer.backend.initialization import LogDirInfo
from runners.viewers.eval_viewer.backend.visualization import create_overlaid_score_map
from runners.viewers.eval_viewer.callbacks.helpers import collect_score_maps
from runners.viewers.eval_viewer.layout.components import (
    create_button_grid,
    create_color_bar,
)


def validate_overlaid_trigger(
    epoch: Optional[int],
    metric_name: Optional[str],
    percentile_drag: Optional[float],
    percentile_value: Optional[float],
) -> None:
    if epoch is None or metric_name is None:
        raise PreventUpdate

    percentile = percentile_drag if percentile_drag is not None else percentile_value
    if percentile is None:
        raise PreventUpdate


def resolve_percentile(
    percentile_drag: Optional[float], percentile_value: Optional[float]
) -> float:
    # Input validations
    assert percentile_drag is None or isinstance(
        percentile_drag, (int, float)
    ), "percentile_drag must be numeric or None"
    assert percentile_value is None or isinstance(
        percentile_value, (int, float)
    ), "percentile_value must be numeric or None"

    percentile = percentile_drag if percentile_drag is not None else percentile_value
    assert percentile is not None, "percentile must be provided"
    return float(percentile)


def build_overlaid_score_map_children(
    epoch: int,
    metric_name: str,
    percentile: float,
    metric_names: List[str],
    num_datapoints: int,
    log_dir_infos: Dict[str, LogDirInfo],
) -> Tuple[html.Div, html.Div]:
    # Input validations
    assert isinstance(epoch, int), f"epoch must be int, got {type(epoch)}"
    assert epoch >= 0, "epoch must be non-negative"
    assert isinstance(
        metric_name, str
    ), f"metric_name must be str, got {type(metric_name)}"
    assert isinstance(
        metric_names, list
    ), f"metric_names must be list, got {type(metric_names)}"
    assert metric_names, "metric_names must be non-empty"
    assert (
        metric_name in metric_names
    ), f"metric_name {metric_name} not found in metrics"
    assert isinstance(
        percentile, (int, float)
    ), f"percentile must be numeric, got {type(percentile)}"
    assert (
        0 <= percentile <= 100
    ), f"percentile must be between 0 and 100, got {percentile}"
    assert isinstance(
        num_datapoints, int
    ), f"num_datapoints must be int, got {type(num_datapoints)}"
    assert num_datapoints > 0, "num_datapoints must be positive"
    assert isinstance(
        log_dir_infos, dict
    ), f"log_dir_infos must be dict, got {type(log_dir_infos)}"
    assert log_dir_infos, "log_dir_infos must be non-empty"

    metric_idx = metric_names.index(metric_name)
    score_maps = collect_score_maps(
        log_dir_infos=log_dir_infos, epoch=epoch, metric_idx=metric_idx
    )
    overlaid_score_map = create_overlaid_score_map(
        score_maps=score_maps, percentile=percentile
    )
    button_grid = create_button_grid(
        num_datapoints=num_datapoints,
        score_map=overlaid_score_map,
        button_type="overlaid-grid-button",
    )
    min_score = np.nanmin(overlaid_score_map)
    max_score = np.nanmax(overlaid_score_map)
    color_bar = create_color_bar(min_score=min_score, max_score=max_score)
    return button_grid, color_bar
