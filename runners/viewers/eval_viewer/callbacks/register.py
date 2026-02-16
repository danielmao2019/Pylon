"""Callback registration for the evaluation viewer."""

from typing import Any, Dict, List

import numpy as np
from dash import Dash

from data.viewer.dataset.backend.backend import DatasetType
from runners.viewers.eval_viewer.backend.initialization import LogDirInfo
from runners.viewers.eval_viewer.callbacks.aggregated.update_aggregated_scores_plot import (
    register_aggregated_scores_plot_callback,
)
from runners.viewers.eval_viewer.callbacks.datapoint.update_datapoint_from_individual import (
    register_datapoint_from_individual_callback,
)
from runners.viewers.eval_viewer.callbacks.datapoint.update_datapoint_from_overlaid import (
    register_datapoint_from_overlaid_callback,
)
from runners.viewers.eval_viewer.callbacks.individual.update_individual_score_maps_from_epoch import (
    register_individual_score_maps_from_epoch_callback,
)
from runners.viewers.eval_viewer.callbacks.individual.update_individual_score_maps_from_metric import (
    register_individual_score_maps_from_metric_callback,
)
from runners.viewers.eval_viewer.callbacks.overlaid.update_overlaid_score_map_from_epoch import (
    register_overlaid_score_map_from_epoch_callback,
)
from runners.viewers.eval_viewer.callbacks.overlaid.update_overlaid_score_map_from_metric import (
    register_overlaid_score_map_from_metric_callback,
)
from runners.viewers.eval_viewer.callbacks.overlaid.update_overlaid_score_map_from_percentile_drag import (
    register_overlaid_score_map_from_percentile_drag_callback,
)
from runners.viewers.eval_viewer.callbacks.overlaid.update_overlaid_score_map_from_percentile_value import (
    register_overlaid_score_map_from_percentile_value_callback,
)
from utils.builders.builder import build_from_config


def register_callbacks(
    app: Dash,
    metric_names: List[str],
    num_datapoints: int,
    log_dir_infos: Dict[str, LogDirInfo],
    per_metric_color_scales: np.ndarray,
    dataset_cfg: Dict[str, Any],
    dataset_type: DatasetType,
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
    assert isinstance(
        dataset_cfg, dict
    ), f"dataset_cfg must be dict, got {type(dataset_cfg)}"
    assert dataset_cfg, "dataset_cfg must be non-empty"
    assert isinstance(
        dataset_type, str
    ), f"dataset_type must be str, got {type(dataset_type)}"

    dataset = build_from_config(dataset_cfg)

    register_aggregated_scores_plot_callback(
        app=app, metric_names=metric_names, log_dir_infos=log_dir_infos
    )
    register_overlaid_score_map_from_metric_callback(
        app=app,
        metric_names=metric_names,
        num_datapoints=num_datapoints,
        log_dir_infos=log_dir_infos,
    )
    register_overlaid_score_map_from_epoch_callback(
        app=app,
        metric_names=metric_names,
        num_datapoints=num_datapoints,
        log_dir_infos=log_dir_infos,
    )
    register_overlaid_score_map_from_percentile_drag_callback(
        app=app,
        metric_names=metric_names,
        num_datapoints=num_datapoints,
        log_dir_infos=log_dir_infos,
    )
    register_overlaid_score_map_from_percentile_value_callback(
        app=app,
        metric_names=metric_names,
        num_datapoints=num_datapoints,
        log_dir_infos=log_dir_infos,
    )
    register_individual_score_maps_from_metric_callback(
        app=app,
        metric_names=metric_names,
        num_datapoints=num_datapoints,
        log_dir_infos=log_dir_infos,
        per_metric_color_scales=per_metric_color_scales,
    )
    register_individual_score_maps_from_epoch_callback(
        app=app,
        metric_names=metric_names,
        num_datapoints=num_datapoints,
        log_dir_infos=log_dir_infos,
        per_metric_color_scales=per_metric_color_scales,
    )
    register_datapoint_from_overlaid_callback(
        app=app,
        dataset=dataset,
        dataset_type=dataset_type,
        log_dir_infos=log_dir_infos,
    )
    register_datapoint_from_individual_callback(
        app=app,
        dataset=dataset,
        dataset_type=dataset_type,
        log_dir_infos=log_dir_infos,
    )
