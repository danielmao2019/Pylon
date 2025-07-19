# LiDAR Simulation Crop Demo

Interactive web-based demonstration of the LiDAR simulation cropping transform. This demo allows you to explore how different sensor poses and parameters affect point cloud filtering in real-time.

## Features

- **Interactive 3D Visualization**: Real-time point cloud visualization with Plotly/Dash
- **Camera Pose Control**: 6DOF interactive camera positioning with sliders
- **Multiple Crop Types**: Range-only, FOV-only, and occlusion-only filtering
- **Point Cloud Variety**: Test with cube, sphere, and scene geometries
- **Real-time Updates**: Immediate feedback as you adjust parameters
- **Fixed Scaling**: Consistent axis ranges and proportions across all visualizations

## Quick Start

Run the interactive web app:

```bash
cd demos/data/transforms/vision_3d/lidar_simulation_crop
python main.py --port 8050
```

Then open http://127.0.0.1:8050 in your web browser.

## Command Line Options

```bash
python main.py --help
```

- `--port PORT`: Web server port (default: 8050)
- `--host HOST`: Web server host (default: 127.0.0.1)
- `--debug`: Enable debug mode for development
- `--no-browser`: Don't automatically open browser

## Usage Instructions

See [webapp/INSTRUCTIONS.md](webapp/INSTRUCTIONS.md) for detailed usage instructions.

## Files

- `main.py`: Main entry point for the web application
- `lidar_crop_demo.py`: Utility functions for point cloud generation
- `webapp/`: Web application modules
  - `backend.py`: Data processing and plot generation
  - `layout.py`: UI layout and control components
  - `callbacks.py`: Interactive functionality
  - `INSTRUCTIONS.md`: Detailed usage instructions

## Related

This demo showcases the `LiDARSimulationCrop` transform located at:
`data/transforms/vision_3d/lidar_simulation_crop.py`
