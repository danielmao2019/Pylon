from typing import List
import dash
from dash import html

from utils.data_loader import validate_log_directories, get_common_metrics
from layouts.main_layout import create_layout
from callbacks.update_plots import register_callbacks


def create_app(log_dirs: List[str]) -> dash.Dash:
    """
    Creates and initializes the Dash application.
    
    Args:
        log_dirs: List of paths to log directories
        
    Returns:
        app: Initialized Dash application
        
    Raises:
        AssertionError: If any validation fails
    """
    # Validate log directories and get max epoch
    max_epoch = validate_log_directories(log_dirs)
    
    # Get common metrics
    metrics = sorted(list(get_common_metrics(log_dirs)))
    
    # Create app
    app = dash.Dash(__name__)
    
    # Create layout
    app.layout = create_layout(max_epoch, metrics, len(log_dirs))
    
    # Register callbacks
    register_callbacks(app, log_dirs)
    
    return app


def run_app(log_dirs: List[str], debug: bool = False, port: int = 8050):
    """
    Runs the Dash application.
    
    Args:
        log_dirs: List of paths to log directories
        debug: Whether to run in debug mode
        port: Port number to run the server on
    """
    app = create_app(log_dirs)
    app.run_server(debug=debug, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dirs", nargs="+", required=True, help="List of log directory paths")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--port", type=int, default=8050, help="Port number")
    args = parser.parse_args()
    
    run_app(log_dirs=args.log_dirs, debug=args.debug, port=args.port)
