#!/usr/bin/env python3
"""
Main entry point for LiDAR simulation cropping visualization web app.
Provides command-line interface for launching the interactive visualization.
"""

import argparse
import sys
import os
import dash

from demos.data.transforms.vision_3d.lidar_simulation_crop.webapp import LiDARVisualizationBackend, LiDARVisualizationLayout, LiDARVisualizationCallbacks


def create_app() -> dash.Dash:
    """Create and configure the Dash application.
    
    Returns:
        Configured Dash application instance
    """
    # Initialize the backend
    backend = LiDARVisualizationBackend()
    
    # Create the Dash app
    app = dash.Dash(__name__)
    app.title = "LiDAR Crop Visualization"
    
    # Initialize layout
    layout_manager = LiDARVisualizationLayout(backend)
    app.layout = layout_manager.create_layout()
    
    # Initialize callbacks
    callback_manager = LiDARVisualizationCallbacks(
        backend=backend,
        control_ids=layout_manager.get_control_ids()
    )
    callback_manager.register_callbacks()
    
    return app


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="LiDAR Simulation Cropping Interactive Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port number for the web server'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host address for the web server'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode for development'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not automatically open browser'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Print startup information
    print("=" * 60)
    print("LiDAR Simulation Cropping - Interactive Visualization")
    print("=" * 60)
    print(f"Starting web server on http://{args.host}:{args.port}/")
    print()
    print("Features:")
    print("  • Interactive 3D visualization of point cloud cropping")
    print("  • Real-time parameter adjustment via slider controls")
    print("  • Support for 3 point cloud types and 3 crop methods")
    print("  • FOV cropping with both Ellipsoid and Frustum modes")
    print("  • Interactive camera pose control with 6DOF sliders")
    print()
    print("Instructions:")
    print("  1. Open the URL above in your web browser")
    print("  2. Use the dropdown controls to select different configurations")
    print("  3. For FOV cropping, choose between Ellipsoid and Frustum modes")
    print("  4. Adjust camera pose sliders for real-time interaction")
    print("  5. Interact with the 3D plot using mouse (rotate, zoom, pan)")
    print("  6. Check the info panel for detailed statistics and configuration")
    print()
    
    if not args.no_browser:
        print("Browser will open automatically...")
    else:
        print("Automatic browser opening disabled.")
    
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        # Create the app
        app = create_app()
        
        # Run the server
        app.run(
            debug=args.debug,
            host=args.host,
            port=args.port,
            dev_tools_hot_reload=args.debug,
            dev_tools_ui=args.debug
        )
        
    except KeyboardInterrupt:
        print("\\nServer stopped by user.")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\\nError: Port {args.port} is already in use.")
            print("Please try a different port using --port argument.")
            print(f"Example: python {sys.argv[0]} --port {args.port + 1}")
        else:
            print(f"\\nError starting server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\\nUnexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()