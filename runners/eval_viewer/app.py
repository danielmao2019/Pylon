from typing import List
import dash
from dash import html
import os
project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
print(project_root)
import sys
sys.path.append(project_root)
os.chdir(project_root)

from runners.eval_viewer.layouts.main_layout import create_layout
from runners.eval_viewer.callbacks.update_plots import register_callbacks
from runners.eval_viewer.backend.data_loader import validate_log_directories, get_common_metrics
from runners.eval_viewer.backend.cache_manager import load_or_create_cache


def create_app(log_dirs: List[str], force_reload: bool = False) -> dash.Dash:
    """
    Creates and initializes the Dash application.

    Args:
        log_dirs: List of paths to log directories
        force_reload: Whether to force recreation of cache

    Returns:
        app: Initialized Dash application

    Raises:
        AssertionError: If any validation fails
    """
    # Load or create cache
    caches = load_or_create_cache(log_dirs, force_reload)

    # Get max epoch and metrics
    max_epoch = validate_log_directories(log_dirs)
    metrics = sorted(list(get_common_metrics(log_dirs)))

    # Create app
    app = dash.Dash(__name__)

    # Create layout
    app.layout = create_layout(max_epoch, metrics, len(log_dirs))

    # Register callbacks
    register_callbacks(app, log_dirs, caches)

    return app


def run_app(log_dirs: List[str], debug: bool = False, port: int = 8050, force_reload: bool = False):
    """
    Runs the Dash application.

    Args:
        log_dirs: List of paths to log directories
        debug: Whether to run in debug mode
        port: Port number to run the server on
        force_reload: Whether to force recreation of cache
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
        '/home/daniel/repos/Pylon/logs/examples/linear',
    ]

    run_app(log_dirs=log_dirs, debug=args.debug, port=args.port, force_reload=args.force_reload)
