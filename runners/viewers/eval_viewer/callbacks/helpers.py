"""Shared helpers for evaluation viewer callbacks."""

from typing import Dict, List

import numpy as np

from runners.viewers.eval_viewer.backend.initialization import LogDirInfo


def collect_score_maps(
    log_dir_infos: Dict[str, LogDirInfo], epoch: int, metric_idx: int
) -> List[np.ndarray]:
    # Input validations
    assert isinstance(
        log_dir_infos, dict
    ), f"log_dir_infos must be dict, got {type(log_dir_infos)}"
    assert log_dir_infos, "log_dir_infos must be non-empty"
    assert isinstance(epoch, int), f"epoch must be int, got {type(epoch)}"
    assert epoch >= 0, "epoch must be non-negative"
    assert isinstance(
        metric_idx, int
    ), f"metric_idx must be int, got {type(metric_idx)}"
    assert metric_idx >= 0, "metric_idx must be non-negative"

    score_maps: List[np.ndarray] = []
    for info in log_dir_infos.values():
        if info.runner_type == "trainer":
            score_maps.append(info.score_map[epoch, metric_idx])
        elif info.runner_type == "evaluator":
            score_maps.append(info.score_map[metric_idx])
        else:
            raise ValueError(f"Unknown runner type: {info.runner_type}")

    assert score_maps, f"No score maps found for metric index {metric_idx}"
    return score_maps
