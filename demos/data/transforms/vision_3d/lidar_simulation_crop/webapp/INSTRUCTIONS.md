# LiDAR Simulation Cropping - Interactive Visualization

## Overview

This interactive web application allows you to explore how different LiDAR sensor configurations affect point cloud filtering. You can adjust camera pose parameters in real-time and see how different crop types filter point clouds.

## Instructions

### Basic Usage
- Use the dropdowns to select different point clouds and crop types
- Use the camera pose sliders to position and orient the sensor interactively
- Use the crop parameter sliders to adjust filtering settings (enabled based on crop type)
- Blue points (transparent) = original points that were removed
- Red points = points kept after cropping
- Black diamond = sensor position
- Green arrow = sensor viewing direction
- Purple surface = range limit (for range-only cropping)
- Orange lines = field of view boundaries (for FOV-only cropping)
- Use mouse to rotate, zoom, and pan the 3D view

### Camera Pose Controls

#### Position Controls
- **Azimuth**: Horizontal rotation around origin (0° = +X axis, 90° = +Y axis)
- **Elevation**: Vertical angle above/below horizon (positive = above, negative = below)
- **Distance**: Distance from origin to sensor position

#### Rotation Controls
- **Yaw**: Sensor rotation around its Z-axis (relative to look-at-origin)
- **Pitch**: Sensor rotation around its Y-axis (tilt up/down)
- **Roll**: Sensor rotation around its X-axis (bank left/right)

### Crop Types

#### Range Only
Filters points based on distance from sensor (adjustable max range). Only the "Range Max" slider is enabled for this mode.

#### FOV Only
Filters points based on field-of-view cone (adjustable horizontal and vertical FOV). The "Horizontal FOV" and "Vertical FOV Span" sliders are enabled for this mode.

#### Occlusion Only
Filters points based on line-of-sight visibility from sensor. No additional parameters are needed for this mode.

### Point Cloud Types

#### Cube
3000 points on surface of 4×4×4 cube - demonstrates geometric edge effects and how cropping affects regular geometric shapes.

#### Sphere
2000 points with uniform volume distribution - shows smooth surface filtering and how spherical geometries respond to different crop types.

#### Scene
4000 points representing complex outdoor environment with multiple objects - demonstrates realistic LiDAR simulation scenarios.

## Tips

- Start with the default settings and gradually adjust one parameter at a time to understand its effect
- Try different camera positions around the point cloud to see how perspective affects filtering
- Compare how the same crop settings affect different point cloud geometries
- Use the statistics panel to quantify the filtering effectiveness
- The coordinate system follows standard conventions: +X forward, +Y left, +Z up in sensor frame

## Performance Notes

- The visualization updates in real-time as you adjust sliders
- Large parameter changes may take a moment to compute
- The web interface is optimized for interactive exploration and real-time feedback