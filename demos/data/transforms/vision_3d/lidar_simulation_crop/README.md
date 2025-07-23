# LiDAR Simulation Crop Demo

Interactive web-based demonstration of the LiDAR simulation cropping transform. This demo allows you to explore how different sensor poses and parameters affect point cloud filtering in real-time.

## Features

- **Interactive 3D Visualization**: Real-time point cloud visualization with Plotly/Dash
- **Camera Pose Control**: 6DOF interactive camera positioning with sliders
- **Multiple Crop Types**: Range-only, FOV-only, and occlusion-only filtering
- **FOV Modes**: Support for both Ellipsoid (LiDAR-style) and Frustum (Camera-style) FOV cropping
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

### Getting Started

1. **Launch the Application**:
   ```bash
   cd demos/data/transforms/vision_3d/lidar_simulation_crop
   python main.py --port 8050
   ```
   The web interface will automatically open in your browser at `http://127.0.0.1:8050`

2. **Basic Controls**:
   - **Point Cloud**: Choose between Cube, Sphere, or Scene geometries
   - **Crop Type**: Select Range Only, FOV Only, or Occlusion Only filtering
   - **Camera Pose**: Use 6DOF sliders (azimuth, elevation, distance, yaw, pitch, roll)

### FOV Cropping Modes

When using "FOV Only" crop type, you can choose between two geometric modes:

- **Ellipsoid (LiDAR-style)**: 
  - Uses spherical coordinates for curved boundaries
  - Typical of spinning LiDAR sensors
  - Supports full 360° horizontal coverage
  - Best for simulating rotating laser scanners

- **Frustum (Camera-style)**:
  - Uses perspective projection for straight-edge pyramidal shapes
  - Matches camera field-of-view geometry
  - Limited to realistic camera ranges (< 180°)
  - Best for pixel-wise correspondence with camera images

### Interactive Features

3. **Real-time Parameter Adjustment**:
   - **Range Cropping**: Adjust maximum range slider (active for Range Only)
   - **FOV Cropping**: Set horizontal and vertical FOV angles (active for FOV Only)
   - **FOV Mode**: Switch between Ellipsoid and Frustum modes for FOV cropping
   - **Camera Pose**: Modify sensor position and orientation with 6DOF sliders

4. **3D Visualization Interaction**:
   - **Rotate**: Click and drag to rotate the 3D view
   - **Zoom**: Use mouse wheel or zoom controls
   - **Pan**: Hold Shift + click and drag to pan
   - **Reset View**: Double-click to reset camera view

5. **Information Panel**:
   - **Statistics**: View point counts and reduction percentages
   - **Camera Pose**: See current sensor position and rotation
   - **Active Filters**: Check which filtering methods are enabled
   - **Configuration**: Review current crop parameters

### Color Coding

- **Blue (transparent)**: Original point cloud
- **Red**: Points kept after cropping
- **Black diamond**: Sensor position
- **Green arrow**: Sensor forward direction
- **Purple**: Range limit visualization (Range Only)
- **Orange**: FOV boundary visualization (FOV Only)

### Tips for Best Results

- Start with default settings and gradually adjust parameters
- Use the FOV mode dropdown to compare Ellipsoid vs Frustum cropping
- Experiment with different camera poses to understand sensor perspectives
- Check the info panel for detailed statistics and configuration summaries
- Use different point cloud types to see how cropping affects various geometries

See [webapp/INSTRUCTIONS.md](webapp/INSTRUCTIONS.md) for additional technical details.

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
