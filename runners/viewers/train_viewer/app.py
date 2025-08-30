from typing import List
import dash

from runners.viewers.train_viewer.layout.main_layout import create_layout
from runners.viewers.train_viewer.callbacks.load_losses import register_callbacks


def create_app() -> dash.Dash:
    """Create the Dash application for training losses viewer.
    
    Returns:
        app: Dash application instance
    """
    # Create app
    app = dash.Dash(__name__)
    
    # Create layout
    app.layout = create_layout()
    
    # Register callbacks
    register_callbacks(app)
    
    return app


def run_app(port: int = 8050) -> None:
    """Run the Dash application.
    
    Args:
        port: Port to run the server on
    """
    assert isinstance(port, int), f"port must be int, got {type(port)}"
    assert 1024 <= port <= 65535, f"port must be between 1024 and 65535, got {port}"
    
    app = create_app()
    app.run(debug=False, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch the training losses viewer")
    parser.add_argument("--port", type=int, default=8050, help="Port number")
    args = parser.parse_args()
    
    # Read log directories from file
    import os
    log_dirs_file = os.path.join(os.path.dirname(__file__), 'log_dirs.txt')
    assert os.path.isfile(log_dirs_file), f"Log directories file must exist: {log_dirs_file}"
    
    with open(log_dirs_file, 'r') as f:
        log_dirs = [line.strip() for line in f.readlines() if line.strip()]
    if log_dirs:
        log_dirs_text = ", ".join(log_dirs)
        print(f"Found log directories in {log_dirs_file}:")
        print(log_dirs_text)
    
    run_app(port=args.port)
