from typing import Optional, List, Tuple
import dash
from .layout import create_layout, initialize_monitor
from .callback import register_callbacks

def create_app(config_files: List[str], expected_files: List[str], epochs: int, gpu_pool: List[Tuple[str, List[int]]]) -> dash.Dash:
    """Create the Dash application.
    
    Args:
        config_files: List of config file paths
        expected_files: List of expected file patterns
        epochs: Total number of epochs
        gpu_pool: List of (server, gpu_indices) tuples
        
    Returns:
        app: Dash application instance
    """
    # Initialize GPU monitor
    initialize_monitor(gpu_pool)
    
    # Create app
    app = dash.Dash(__name__)
    
    # Create layout
    app.layout = create_layout(config_files, expected_files, epochs)
    
    # Register callbacks
    register_callbacks(app, config_files, expected_files, epochs)
    
    return app

def launch_app(app: dash.Dash, port: Optional[int] = 8050) -> None:
    """Launch the dashboard.
    
    Args:
        app: Dash application instance
        port: Port to run the server on
    """
    app.run(debug=True, port=port)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--config-files", nargs="+", required=True, help="List of config file paths")
    parser.add_argument("--expected-files", nargs="+", required=True, help="List of expected file patterns")
    parser.add_argument("--epochs", type=int, required=True, help="Total number of epochs")
    parser.add_argument("--gpu-pool", nargs="+", required=True, help="List of GPU pool entries in format 'server:gpu_indices' (e.g. 'server1:0,1,2 server2:0,1')")
    args = parser.parse_args()
    
    # Parse GPU pool
    gpu_pool = []
    for entry in args.gpu_pool:
        server, indices_str = entry.split(':')
        indices = [int(idx) for idx in indices_str.split(',')]
        gpu_pool.append((server, indices))
    
    app = create_app(
        config_files=args.config_files,
        expected_files=args.expected_files,
        epochs=args.epochs,
        gpu_pool=gpu_pool
    )
    launch_app(app, port=args.port)
