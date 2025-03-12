#!/usr/bin/env python3
"""
Command line interface for the dataset viewer.

This script provides a simple command-line entry point to launch the dataset viewer.
"""
import argparse
import os
import sys

# Add repository root to Python path
repo_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data.viewer import run_viewer


def main():
    """Run the dataset viewer application."""
    parser = argparse.ArgumentParser(description="Dataset Viewer CLI")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Print debug information
    print(f"Current working directory: {os.getcwd()}")
    print(f"Repository root (estimated): {repo_root}")
    print(f"Python sys.path: {sys.path}")
    
    # Run the viewer
    run_viewer(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
