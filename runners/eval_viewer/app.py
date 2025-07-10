from typing import List
import dash
import os
project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
print(project_root)
import sys
sys.path.append(project_root)
os.chdir(project_root)

from runners.eval_viewer.layout.main_layout import create_layout
from runners.eval_viewer.callbacks.update_plots import register_callbacks
from runners.eval_viewer.callbacks.datapoint_viewer import register_datapoint_viewer_callbacks
from runners.eval_viewer.backend.initialization import initialize_log_dirs


def create_app(log_dirs: List[str], force_reload: bool = False) -> dash.Dash:
    """Create the Dash application.

    Args:
        log_dirs: List of paths to log directories
        force_reload: Whether to force reload of cached data

    Returns:
        app: Dash application instance
    """
    # Initialize log directories
    max_epochs, metric_names, num_datapoints, dataset_cfg, dataset_type, log_dir_infos, per_metric_color_scales = initialize_log_dirs(log_dirs, force_reload)

    # Extract run names from log directories
    run_names = [os.path.basename(os.path.normpath(log_dir)) for log_dir in log_dirs]

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
    app = create_app(log_dirs, force_reload)
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
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
