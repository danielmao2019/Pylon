from typing import List
import dash
import os

from runners.viewers.eval_viewer.layout.main_layout import create_layout
from runners.viewers.eval_viewer.callbacks.update_plots import register_callbacks
from runners.viewers.eval_viewer.callbacks.datapoint_viewer import register_datapoint_viewer_callbacks
from runners.viewers.eval_viewer.backend.initialization import initialize_log_dirs


def create_app(log_dirs: List[str], force_reload: bool = False) -> dash.Dash:
    """Create the Dash application.

    Args:
        log_dirs: List of paths to log directories
        force_reload: Whether to force reload of cached data

    Returns:
        app: Dash application instance
    """
    # Input validation following CLAUDE.md fail-fast patterns
    assert log_dirs is not None, "log_dirs must not be None"
    assert isinstance(log_dirs, list), f"log_dirs must be list, got {type(log_dirs)}"
    assert len(log_dirs) > 0, f"log_dirs must not be empty"
    assert all(isinstance(log_dir, str) for log_dir in log_dirs), f"All log_dirs must be strings, got {log_dirs}"
    assert all(os.path.exists(log_dir) for log_dir in log_dirs), f"All log directories must exist, got {log_dirs}"
    
    assert isinstance(force_reload, bool), f"force_reload must be bool, got {type(force_reload)}"
    # Initialize log directories
    max_epochs, metric_names, num_datapoints, dataset_cfg, dataset_type, log_dir_infos, per_metric_color_scales = initialize_log_dirs(log_dirs, force_reload)

    # Extract experiment names from aggregated log_dir_infos (now keyed by experiment name)
    run_names = list(log_dir_infos.keys())

    # Create app
    app = dash.Dash(__name__)

    # Create layout
    app.layout = create_layout(max_epochs, metric_names, run_names)

    # Register callbacks
    register_callbacks(app, metric_names, num_datapoints, log_dir_infos, per_metric_color_scales)
    register_datapoint_viewer_callbacks(app, dataset_cfg, dataset_type, log_dir_infos)
    return app


def run_app(log_dirs: List[str], force_reload: bool = False, debug: bool = True, port: int = 8050):
    """Run the Dash application.

    Args:
        log_dirs: List of paths to log directories
        force_reload: Whether to force reload of cached data
        debug: Whether to run in debug mode
        port: Port to run the server on
    """
    # Input validation following CLAUDE.md fail-fast patterns
    assert log_dirs is not None, "log_dirs must not be None"
    assert isinstance(log_dirs, list), f"log_dirs must be list, got {type(log_dirs)}"
    assert len(log_dirs) > 0, f"log_dirs must not be empty"
    assert all(isinstance(log_dir, str) for log_dir in log_dirs), f"All log_dirs must be strings, got {log_dirs}"
    
    assert isinstance(force_reload, bool), f"force_reload must be bool, got {type(force_reload)}"
    assert isinstance(debug, bool), f"debug must be bool, got {type(debug)}"
    assert isinstance(port, int), f"port must be int, got {type(port)}"
    assert 1024 <= port <= 65535, f"port must be between 1024 and 65535, got {port}"
    app = create_app(log_dirs, force_reload)
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch the evaluation results viewer")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--port", type=int, default=8050, help="Port number")
    parser.add_argument("--force-reload", action="store_true", help="Force recreation of cache")
    args = parser.parse_args()

    log_dirs = [
        "./logs/benchmarks/point_cloud_registration/kitti/ICP_run_0",
        "./logs/benchmarks/point_cloud_registration/kitti/RANSAC_FPFH_run_0",
        "./logs/benchmarks/point_cloud_registration/kitti/TeaserPlusPlus_run_0",
    ]

    run_app(log_dirs=log_dirs, debug=args.debug, port=args.port, force_reload=args.force_reload)
