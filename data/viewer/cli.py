#!/usr/bin/env python3
"""
Command line interface for the dataset viewer.

This script provides a simple command-line entry point to launch the dataset viewer.
"""
import argparse
import os
import sys
from pathlib import Path

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data.viewer.viewer import DatasetViewer


def main() -> None:
    """Run the dataset viewer application."""
    parser = argparse.ArgumentParser(description="Dataset Viewer CLI")
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind to")

    args = parser.parse_args()

    # Print debug information
    print(f"Current working directory: {os.getcwd()}")
    print(f"Repository root (estimated): {REPO_ROOT}")
    print(f"Python sys.path: {sys.path}")

    # Create and run the viewer
    viewer = DatasetViewer()
    viewer.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
