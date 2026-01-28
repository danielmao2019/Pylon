"""Helpers for the individual score map callbacks."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from dash import html

from runners.viewers.eval_viewer.backend.initialization import LogDirInfo
from runners.viewers.eval_viewer.callbacks.helpers import collect_score_maps
from runners.viewers.eval_viewer.layout.components import (
    create_button_grid,
    create_color_bar,
)


def create_grid_and_colorbar(
    score_map: np.ndarray,
    run_idx: int,
    num_datapoints: int,
    min_score: float,
    max_score: float,
) -> Tuple[int, List[html.Div]]:
    # Input validations
    assert isinstance(
        score_map, np.ndarray
    ), f"score_map must be np.ndarray, got {type(score_map)}"
    assert score_map.ndim == 2, f"score_map must be 2D, got shape {score_map.shape}"
    assert isinstance(run_idx, int), f"run_idx must be int, got {type(run_idx)}"
    assert run_idx >= 0, "run_idx must be non-negative"
    assert isinstance(
        num_datapoints, int
    ), f"num_datapoints must be int, got {type(num_datapoints)}"
    assert num_datapoints > 0, "num_datapoints must be positive"
    assert isinstance(
        min_score, (int, float)
    ), f"min_score must be numeric, got {type(min_score)}"
    assert isinstance(
        max_score, (int, float)
    ), f"max_score must be numeric, got {type(max_score)}"
    assert max_score >= min_score, "max_score must be >= min_score"

    button_grid = create_button_grid(
        num_datapoints=num_datapoints,
        score_map=score_map,
        button_type="individual-grid-button",
        run_idx=run_idx,
        min_score=min_score,
        max_score=max_score,
    )
    color_bar = create_color_bar(min_score=min_score, max_score=max_score)
    return run_idx, [button_grid, color_bar]


def build_individual_score_map_children(
    epoch: int,
    metric_name: str,
    metric_names: List[str],
    num_datapoints: int,
    log_dir_infos: Dict[str, LogDirInfo],
    per_metric_color_scales: np.ndarray,
) -> List[html.Div]:
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

    metric_idx = metric_names.index(metric_name)
    score_maps = collect_score_maps(
        log_dir_infos=log_dir_infos, epoch=epoch, metric_idx=metric_idx
    )
    min_score, max_score = per_metric_color_scales[metric_idx]

    results: List[List[html.Div]] = [None] * len(score_maps)
    with ThreadPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(
                create_grid_and_colorbar,
                score_map=score_map,
                run_idx=i,
                num_datapoints=num_datapoints,
                min_score=min_score,
                max_score=max_score,
            ): i
            for i, score_map in enumerate(score_maps)
        }

        for future in as_completed(future_to_idx):
            run_idx, grid_and_bar = future.result()
            results[run_idx] = grid_and_bar

    return [item for sublist in results for item in sublist]
