"""Entry point for the evaluation viewer."""

import argparse
from pathlib import Path
from typing import List

from dash import Dash

from runners.viewers.eval_viewer.backend.initialization import initialize_log_dirs
from runners.viewers.eval_viewer.callbacks.register import register_callbacks
from runners.viewers.eval_viewer.layout.components import build_layout


def create_app(log_dirs: List[str], force_reload: bool = False) -> Dash:
    # Input validations
    assert log_dirs is not None, "log_dirs must not be None"
    assert isinstance(log_dirs, list), f"log_dirs must be list, got {type(log_dirs)}"
    assert log_dirs, "log_dirs must be non-empty"
    assert all(
        isinstance(log_dir, str) for log_dir in log_dirs
    ), "log_dirs must contain strings"
    assert all(
        Path(log_dir).exists() for log_dir in log_dirs
    ), "all log_dirs must exist"
    assert isinstance(
        force_reload, bool
    ), f"force_reload must be bool, got {type(force_reload)}"

    (
        max_epochs,
        metric_names,
        num_datapoints,
        dataset_cfg,
        dataset_type,
        log_dir_infos,
        per_metric_color_scales,
    ) = initialize_log_dirs(log_dirs=log_dirs, force_reload=force_reload)
    run_names = list(log_dir_infos.keys())

    app = Dash(__name__)
    build_layout(
        app=app,
        max_epochs=max_epochs,
        metric_names=metric_names,
        run_names=run_names,
    )
    register_callbacks(
        app=app,
        metric_names=metric_names,
        num_datapoints=num_datapoints,
        log_dir_infos=log_dir_infos,
        per_metric_color_scales=per_metric_color_scales,
        dataset_cfg=dataset_cfg,
        dataset_type=dataset_type,
    )
    return app


def run_app(log_dirs: List[str], force_reload: bool = False, port: int = 8050) -> None:
    # Input validations
    assert isinstance(port, int), f"port must be int, got {type(port)}"
    assert 1024 <= port <= 65535, f"port must be between 1024 and 65535, got {port}"

    app = create_app(log_dirs=log_dirs, force_reload=force_reload)
    app.run(host="0.0.0.0", port=port, debug=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the evaluation results viewer")
    parser.add_argument("--port", type=int, default=8050, help="Port number")
    parser.add_argument(
        "--force-reload", action="store_true", help="Force recreation of cache"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    log_dirs = [
        "./logs/benchmarks/point_cloud_registration/kitti/ICP_run_0",
        "./logs/benchmarks/point_cloud_registration/kitti/RANSAC_FPFH_run_0",
        "./logs/benchmarks/point_cloud_registration/kitti/TeaserPlusPlus_run_0",
    ]

    run_app(log_dirs=log_dirs, force_reload=args.force_reload, port=args.port)


if __name__ == "__main__":
    main()
