import argparse
import os
import sys

# Add repository root to Python path
repo_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from runners.viewer.viewer import TrainerViewer


def main():
    parser = argparse.ArgumentParser(description="Training Visualization Tool")
    parser.add_argument(
        "--config-filepath",
        type=str,
        required=True,
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address for the web server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port number for the web server"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Initialize and run the viewer
    viewer = TrainerViewer(args.config_filepath)
    viewer.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
