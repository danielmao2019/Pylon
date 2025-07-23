#!/usr/bin/env python3
"""
Backend module for LiDAR simulation cropping visualization.
Handles data generation, processing, and 3D plot creation.
"""

import plotly.graph_objects as go
import numpy as np
import torch
from typing import Dict, List, Tuple, Any

from demos.data.transforms.vision_3d.lidar_simulation_crop.webapp.backend.point_cloud_utils import create_toy_point_cloud
from data.transforms.vision_3d import LiDARSimulationCrop


class LiDARVisualizationBackend:
    """Backend class for handling LiDAR visualization data and processing."""
    
    def __init__(self):
        """Initialize the backend."""
        self.crop_configs = self._create_crop_configs()
        self.axis_ranges = self._calculate_fixed_axis_ranges()
    
    def _create_crop_configs(self) -> Dict[str, LiDARSimulationCrop]:
        """Create crop configuration objects."""
        return {
            'range_only': LiDARSimulationCrop(
                max_range=6.0,
                fov=(360.0, 180.0),  # Default wide FOV since not used
                ray_density_factor=0.8,
                apply_range_filter=True,
                apply_fov_filter=False,
                apply_occlusion_filter=False
            ),
            'fov_only': LiDARSimulationCrop(
                max_range=100.0,  # Very large range so no range filtering
                fov=(80.0, 40.0),  # (horizontal_fov, vertical_fov)
                ray_density_factor=0.8,
                apply_range_filter=False,
                apply_fov_filter=True,
                apply_occlusion_filter=False
            ),
            'occlusion_only': LiDARSimulationCrop(
                max_range=100.0,  # Very large range so no range filtering
                fov=(360.0, 180.0),  # Full circle and sphere so no FOV filtering
                ray_density_factor=0.8,
                apply_range_filter=False,
                apply_fov_filter=False,
                apply_occlusion_filter=True
            )
        }
    
    def get_point_cloud(self, cloud_name: str) -> torch.Tensor:
        """Get point cloud by name.
        
        Args:
            cloud_name: Name of point cloud ('cube', 'sphere', 'scene')
            
        Returns:
            Point cloud tensor [N, 3]
        """
        if cloud_name == 'cube':
            return create_toy_point_cloud('cube', 3000, seed=42)
        elif cloud_name == 'sphere':
            return create_toy_point_cloud('sphere', 2000, seed=43)
        elif cloud_name == 'scene':
            return create_toy_point_cloud('scene', 4000, seed=44)
        else:
            raise ValueError(f"Unknown cloud type: {cloud_name}")
    
    def _calculate_fixed_axis_ranges(self) -> Dict[str, List[float]]:
        """Calculate fixed axis ranges for consistent scaling across all point clouds.
        
        Returns:
            Dictionary with 'x', 'y', 'z' keys and [min, max] ranges
        """
        # Get all point clouds to determine overall spatial extent
        all_clouds = []
        for cloud_name in ['cube', 'sphere', 'scene']:
            cloud = self.get_point_cloud(cloud_name)
            all_clouds.append(cloud.numpy())
        
        # Combine all point clouds
        combined_points = np.vstack(all_clouds)
        
        # Calculate overall min/max for each axis
        x_min, x_max = combined_points[:, 0].min(), combined_points[:, 0].max()
        y_min, y_max = combined_points[:, 1].min(), combined_points[:, 1].max()
        z_min, z_max = combined_points[:, 2].min(), combined_points[:, 2].max()
        
        # Add padding (10% on each side)
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        x_padding = x_range * 0.1
        y_padding = y_range * 0.1
        z_padding = z_range * 0.1
        
        x_min_padded = x_min - x_padding
        x_max_padded = x_max + x_padding
        y_min_padded = y_min - y_padding
        y_max_padded = y_max + y_padding
        z_min_padded = z_min - z_padding
        z_max_padded = z_max + z_padding
        
        # Make ranges symmetric around origin and equal scale
        max_range = max(
            x_max_padded - x_min_padded,
            y_max_padded - y_min_padded,
            z_max_padded - z_min_padded
        )
        
        # Center ranges around origin with equal scale
        half_range = max_range / 2
        
        return {
            'x': [-half_range, half_range],
            'y': [-half_range, half_range],
            'z': [-half_range, half_range]
        }
    
    def create_camera_pose(self, azimuth: float, elevation: float, distance: float, 
                          yaw: float, pitch: float, roll: float) -> torch.Tensor:
        """Create camera pose from spherical coordinates and rotations.
        
        Camera Control Convention:
        - Azimuth: Horizontal rotation around world Z-axis (0° = +X, 90° = +Y)
        - Elevation: Vertical angle from horizon (0° = horizontal, +90° = up, -90° = down) 
        - Distance: Distance from origin
        - Yaw: Camera left/right rotation around its local Z-axis (looking direction)
        - Pitch: Camera up/down rotation around its local Y-axis (tilt)
        - Roll: Camera twist rotation around its local X-axis (forward direction)
        
        Args:
            azimuth: Horizontal angle from +X axis in degrees [0, 360]
            elevation: Vertical angle from horizon in degrees [-90, 90]
            distance: Distance from origin [1, 20]
            yaw: Camera yaw rotation in degrees [-180, 180] (left/right turn)
            pitch: Camera pitch rotation in degrees [-90, 90] (up/down tilt)
            roll: Camera roll rotation in degrees [-180, 180] (twist)
            
        Returns:
            4x4 extrinsics matrix (sensor-to-world transformation)
        """
        # Convert to radians
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        
        # Step 1: Convert spherical coordinates to camera position
        camera_x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        camera_y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        camera_z = distance * np.sin(elevation_rad)
        camera_pos = np.array([camera_x, camera_y, camera_z])
        
        # Step 2: Create initial look-at orientation (camera looking toward origin)
        # World coordinate system: +X right, +Y forward, +Z up  
        # Sensor coordinate system: +X forward, +Y left, +Z up (matches LiDAR crop expectation)
        forward = -camera_pos / np.linalg.norm(camera_pos)  # Point toward origin
        world_up = np.array([0.0, 0.0, 1.0])  # World up is +Z
        
        # Handle gimbal lock case when looking straight up/down
        if np.abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([1.0, 0.0, 0.0])  # Use +X as up when looking vertically
        
        # Compute camera coordinate system vectors in world frame
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Step 3: Apply additional rotations in camera's local coordinate system
        # Sensor frame: +X forward, +Y left, +Z up (matches LiDAR crop convention)
        
        # Create rotation matrices for each axis (in sensor's local frame)
        # Roll: rotation around forward axis (+X)
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])
        
        # Pitch: rotation around left axis (+Y) 
        R_pitch = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        
        # Yaw: rotation around up axis (+Z)
        R_yaw = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        # Combine rotations: Roll, then Pitch, then Yaw (intrinsic rotations)
        local_rotation = R_yaw @ R_pitch @ R_roll
        
        # Step 4: Transform local rotation to world coordinates
        # Base sensor orientation matrix (sensor-to-world)
        # Sensor frame: [+X forward, +Y left, +Z up] → World frame
        sensor_to_world_rotation = np.column_stack([forward, -right, up])  # [forward, left, up] as columns
        
        # Apply additional rotations in sensor's local frame
        final_sensor_to_world_rotation = sensor_to_world_rotation @ local_rotation
        
        # Step 5: Create 4x4 extrinsics matrix (sensor-to-world)
        extrinsics = torch.eye(4, dtype=torch.float32)
        extrinsics[:3, :3] = torch.tensor(final_sensor_to_world_rotation, dtype=torch.float32)  # sensor-to-world rotation
        extrinsics[:3, 3] = torch.tensor(camera_pos, dtype=torch.float32)  # sensor position
        
        return extrinsics

    def get_crop_config(self, crop_type: str, **params) -> LiDARSimulationCrop:
        """Get crop configuration by type with dynamic parameters.
        
        Args:
            crop_type: Type of cropping ('range_only', 'fov_only', 'occlusion_only')
            **params: Dynamic parameters for crop configuration
                - range_max: Maximum range for range_only cropping
                - h_fov: Horizontal FOV for fov_only cropping
                - v_fov: Vertical FOV for fov_only cropping
                - fov_mode: FOV mode for fov_only cropping ('ellipsoid' or 'frustum')
            
        Returns:
            LiDARSimulationCrop configuration object
        """
        if crop_type == 'range_only':
            range_max = params.get('range_max', 6.0)
            return LiDARSimulationCrop(
                max_range=range_max,
                fov=(360.0, 180.0),  # Default wide FOV since not used
                ray_density_factor=0.8,
                apply_range_filter=True,
                apply_fov_filter=False,
                apply_occlusion_filter=False
            )
        elif crop_type == 'fov_only':
            h_fov = params.get('h_fov', 80.0)
            v_fov = params.get('v_fov', 40.0)  # Total span around center (0)
            fov_mode = params.get('fov_mode', 'ellipsoid')  # Default to ellipsoid mode
            return LiDARSimulationCrop(
                max_range=100.0,  # Very large range so no range filtering
                fov=(h_fov, v_fov),  # (horizontal_fov, vertical_fov)
                fov_crop_mode=fov_mode,  # ellipsoid or frustum
                ray_density_factor=0.8,
                apply_range_filter=False,
                apply_fov_filter=True,
                apply_occlusion_filter=False
            )
        elif crop_type == 'occlusion_only':
            return LiDARSimulationCrop(
                max_range=100.0,  # Very large range so no range filtering
                fov=(360.0, 180.0),  # Full circle and sphere so no FOV filtering
                ray_density_factor=0.8,
                apply_range_filter=False,
                apply_fov_filter=False,
                apply_occlusion_filter=True
            )
        else:
            raise ValueError(f"Unknown crop type: {crop_type}")
    
    def process_cropping(self, cloud_name: str, crop_type: str, 
                        azimuth: float, elevation: float, distance: float,
                        yaw: float, pitch: float, roll: float, **crop_params: Any) -> Dict[str, Any]:
        """Process point cloud cropping for given configuration.
        
        Args:
            cloud_name: Name of point cloud ('cube', 'sphere', 'scene')
            crop_type: Type of cropping ('range_only', 'fov_only', 'occlusion_only')
            azimuth: Horizontal angle from +X axis in degrees [0, 360]
            elevation: Vertical angle from horizon in degrees [-90, 90]
            distance: Distance from origin [1, 20]
            yaw: Camera yaw rotation in degrees [-180, 180]
            pitch: Camera pitch rotation in degrees [-90, 90]
            roll: Camera roll rotation in degrees [-180, 180]
            **crop_params: Dynamic crop parameters (range_max, h_fov, v_fov, fov_mode)
            
        Returns:
            Dictionary with processed data including original points, cropped points,
            sensor information, and metadata
        """
        # Get data
        original_points = self.get_point_cloud(cloud_name)
        crop_config = self.get_crop_config(crop_type, **crop_params)
        
        # Create camera pose from interactive controls
        sensor_extrinsics = self.create_camera_pose(azimuth, elevation, distance, yaw, pitch, roll)
        
        # Apply cropping
        pc: Dict[str, torch.Tensor] = {'pos': original_points, 'feat': torch.ones(len(original_points), 1)}
        cropped_pc: Dict[str, torch.Tensor] = crop_config._call_single(pc, sensor_extrinsics, generator=torch.Generator())
        cropped_points: torch.Tensor = cropped_pc['pos']
        
        # Extract sensor info
        sensor_pos = sensor_extrinsics[:3, 3].numpy()
        sensor_rot = sensor_extrinsics[:3, :3].numpy()
        
        # Calculate reduction percentage
        reduction = (1 - len(cropped_points) / len(original_points)) * 100
        
        # Create pose description for display
        pose_description = f"Az:{azimuth:.0f}° El:{elevation:.0f}° Dist:{distance:.1f}m Y:{yaw:.0f}° P:{pitch:.0f}° R:{roll:.0f}°"
        
        return {
            'original_points': original_points.numpy(),
            'cropped_points': cropped_points.numpy(),
            'sensor_pos': sensor_pos,
            'sensor_rot': sensor_rot,
            'crop_config': crop_config,
            'pose_description': pose_description,
            'reduction': reduction,
            'camera_params': {
                'azimuth': azimuth,
                'elevation': elevation,
                'distance': distance,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll
            },
            'crop_params': crop_params
        }
    
    def create_3d_scatter_plot(self, cloud_name: str, crop_type: str, 
                              azimuth: float, elevation: float, distance: float,
                              yaw: float, pitch: float, roll: float, **crop_params: Any) -> go.Figure:
        """Create a 3D scatter plot for the specified configuration.
        
        Args:
            cloud_name: Name of point cloud
            crop_type: Type of cropping
            azimuth: Horizontal angle from +X axis in degrees
            elevation: Vertical angle from horizon in degrees
            distance: Distance from origin
            yaw: Camera yaw rotation in degrees
            pitch: Camera pitch rotation in degrees
            roll: Camera roll rotation in degrees
            **crop_params: Dynamic crop parameters (including fov_mode for fov_only)
            
        Returns:
            Plotly Figure object
        """
        try:
            # Process the data
            data = self.process_cropping(cloud_name, crop_type, azimuth, elevation, distance, 
                                       yaw, pitch, roll, **crop_params)
            
            original_np = data['original_points']
            cropped_np = data['cropped_points']
            sensor_pos = data['sensor_pos']
            sensor_rot = data['sensor_rot']
            crop_config = data['crop_config']
            pose_description = data['pose_description']
            reduction = data['reduction']
            
            # Create figure
            fig = go.Figure()
            
            # Add original points (transparent blue)
            fig.add_trace(go.Scatter3d(
                x=original_np[:, 0],
                y=original_np[:, 1], 
                z=original_np[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='blue',
                    opacity=0.1
                ),
                name='Original Points',
                showlegend=True
            ))
            
            # Add kept points (red)
            fig.add_trace(go.Scatter3d(
                x=cropped_np[:, 0],
                y=cropped_np[:, 1],
                z=cropped_np[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color='red',
                    opacity=0.8
                ),
                name=f'Kept Points ({len(cropped_np):,})',
                showlegend=True
            ))
            
            # Add sensor position
            fig.add_trace(go.Scatter3d(
                x=[sensor_pos[0]],
                y=[sensor_pos[1]],
                z=[sensor_pos[2]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='black',
                    symbol='diamond'
                ),
                name='Sensor Position',
                showlegend=True
            ))
            
            # Add sensor orientation arrow
            forward_dir = sensor_rot[:, 0]  # Forward direction
            arrow_length = 3.0
            arrow_end = sensor_pos + forward_dir * arrow_length
            
            fig.add_trace(go.Scatter3d(
                x=[sensor_pos[0], arrow_end[0]],
                y=[sensor_pos[1], arrow_end[1]],
                z=[sensor_pos[2], arrow_end[2]],
                mode='lines+markers',
                line=dict(color='green', width=8),
                marker=dict(size=[8, 12], color=['green', 'green']),
                name='Sensor Direction',
                showlegend=True
            ))
            
            # Add crop-specific visualizations
            if crop_type == 'range_only' and crop_config.apply_range_filter:
                self._add_range_visualization(fig, sensor_pos, crop_config.max_range)
                
            elif crop_type == 'fov_only' and crop_config.apply_fov_filter:
                fov_mode = crop_params.get('fov_mode', 'ellipsoid')
                self._add_fov_visualization(fig, sensor_pos, sensor_rot, crop_config, fov_mode)
            
            # Create title with FOV mode information
            title_parts = [f"{cloud_name.title()} - {crop_type.replace('_', ' ').title()}"]
            if crop_type == 'fov_only' and 'fov_mode' in crop_params:
                fov_mode = crop_params['fov_mode']
                title_parts[0] += f" ({fov_mode.title()})"
            
            # Set layout with fixed axis ranges for consistent scaling
            fig.update_layout(
                title=f"{title_parts[0]}<br>"
                      f"<sub>{pose_description} | Points: {len(original_np):,} → {len(cropped_np):,} ({reduction:.1f}% reduction)</sub>",
                scene=dict(
                    xaxis=dict(
                        title="X",
                        range=self.axis_ranges['x']
                    ),
                    yaxis=dict(
                        title="Y", 
                        range=self.axis_ranges['y']
                    ),
                    zaxis=dict(
                        title="Z",
                        range=self.axis_ranges['z']
                    ),
                    aspectmode='cube'  # Forces equal scaling on all axes
                ),
                showlegend=True,
                margin=dict(l=0, r=0, t=60, b=0),
                uirevision='preserve_view'  # Preserve user's camera view/zoom between updates
            )
            
            return fig
            
        except Exception as e:
            # Return error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error: {str(e)}",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(title="Error in Visualization")
            return fig
    
    def _add_range_visualization(self, fig: go.Figure, sensor_pos: np.ndarray, max_range: float) -> None:
        """Add range sphere visualization to the figure.
        
        Args:
            fig: Plotly figure to add to
            sensor_pos: Sensor position [3]
            max_range: Maximum range value
        """
        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = max_range * np.outer(np.cos(u), np.sin(v)) + sensor_pos[0]
        y_sphere = max_range * np.outer(np.sin(u), np.sin(v)) + sensor_pos[1]
        z_sphere = max_range * np.outer(np.ones(np.size(u)), np.cos(v)) + sensor_pos[2]
        
        # Add sphere surface (wireframe style)
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.2,
            colorscale=[[0, 'purple'], [1, 'purple']],
            showscale=False,
            name='Range Limit',
            hoverinfo='skip'
        ))
    
    def _add_fov_visualization(self, fig: go.Figure, sensor_pos: np.ndarray, 
                             sensor_rot: np.ndarray, crop_config: LiDARSimulationCrop, fov_mode: str = 'ellipsoid') -> None:
        """Add FOV visualization to the figure.
        
        Args:
            fig: Plotly figure to add to
            sensor_pos: Sensor position [3]
            sensor_rot: Sensor rotation matrix [3x3]
            crop_config: LiDAR configuration object
            fov_mode: FOV mode ('ellipsoid' or 'frustum')
        """
        if fov_mode == 'ellipsoid':
            self._add_ellipsoid_fov_visualization(fig, sensor_pos, sensor_rot, crop_config)
        elif fov_mode == 'frustum':
            self._add_frustum_fov_visualization(fig, sensor_pos, sensor_rot, crop_config)
    
    def _add_ellipsoid_fov_visualization(self, fig: go.Figure, sensor_pos: np.ndarray, 
                                       sensor_rot: np.ndarray, crop_config: LiDARSimulationCrop) -> None:
        """Add ellipsoidal FOV visualization to the figure.
        
        Creates a proper ellipsoidal surface visualization by drawing multiple curved
        cross-sections that show the convex ellipsoidal shape.
        
        Args:
            fig: Plotly figure to add to
            sensor_pos: Sensor position [3]
            sensor_rot: Sensor rotation matrix [3x3]
            crop_config: LiDAR configuration object
        """
        horizontal_fov, vertical_fov = crop_config.fov
        h_fov_half = horizontal_fov / 2
        v_fov_half = vertical_fov / 2
        
        frustum_length = 8.0
        
        # Generate ellipsoidal surface using parametric equations
        # θ (theta): azimuth angle [-h_fov_half, +h_fov_half]
        # φ (phi): elevation angle [-v_fov_half, +v_fov_half]
        
        # Create boundary curves showing ellipsoidal shape
        num_curve_points = 20
        num_cross_sections = 5
        
        # 1. Draw horizontal cross-sections at different elevation angles
        elevation_angles = np.linspace(-v_fov_half, v_fov_half, num_cross_sections)
        for i, elev_deg in enumerate(elevation_angles):
            elev_rad = np.radians(elev_deg)
            
            # Create horizontal curve at this elevation
            azimuth_angles = np.linspace(-h_fov_half, h_fov_half, num_curve_points)
            curve_points = []
            
            for azim_deg in azimuth_angles:
                azim_rad = np.radians(azim_deg)
                
                # Spherical to Cartesian conversion for ellipsoidal surface
                # Radius varies with angle to create ellipsoidal shape
                radius = frustum_length * np.cos(elev_rad)  # Ellipsoidal scaling
                
                # Point in sensor coordinates (forward = +X)
                x = frustum_length * np.cos(elev_rad) * np.cos(azim_rad)
                y = frustum_length * np.cos(elev_rad) * np.sin(azim_rad)
                z = frustum_length * np.sin(elev_rad)
                
                # Transform to world coordinates
                local_point = np.array([x, y, z])
                world_point = sensor_rot @ local_point + sensor_pos
                curve_points.append(world_point)
            
            curve_points = np.array(curve_points)
            
            # Use different line styles for different cross-sections
            is_edge = (i == 0 or i == num_cross_sections - 1)
            line_width = 3 if is_edge else 2
            line_color = 'orange' if is_edge else 'gold'
            opacity = 1.0 if is_edge else 0.6
            
            fig.add_trace(go.Scatter3d(
                x=curve_points[:, 0],
                y=curve_points[:, 1],
                z=curve_points[:, 2],
                mode='lines',
                line=dict(color=line_color, width=line_width),
                opacity=opacity,
                name='Ellipsoid Boundary' if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo='skip'
            ))
        
        # 2. Draw vertical cross-sections at different azimuth angles  
        azimuth_angles = np.linspace(-h_fov_half, h_fov_half, num_cross_sections)
        for i, azim_deg in enumerate(azimuth_angles):
            azim_rad = np.radians(azim_deg)
            
            # Create vertical curve at this azimuth
            elevation_angles = np.linspace(-v_fov_half, v_fov_half, num_curve_points)
            curve_points = []
            
            for elev_deg in elevation_angles:
                elev_rad = np.radians(elev_deg)
                
                # Point in sensor coordinates (forward = +X)
                x = frustum_length * np.cos(elev_rad) * np.cos(azim_rad)
                y = frustum_length * np.cos(elev_rad) * np.sin(azim_rad)
                z = frustum_length * np.sin(elev_rad)
                
                # Transform to world coordinates
                local_point = np.array([x, y, z])
                world_point = sensor_rot @ local_point + sensor_pos
                curve_points.append(world_point)
            
            curve_points = np.array(curve_points)
            
            # Use different line styles for edge curves
            is_edge = (i == 0 or i == num_cross_sections - 1)
            line_width = 3 if is_edge else 1
            line_color = 'darkorange' if is_edge else 'gold'
            opacity = 1.0 if is_edge else 0.4
            
            fig.add_trace(go.Scatter3d(
                x=curve_points[:, 0],
                y=curve_points[:, 1],
                z=curve_points[:, 2],
                mode='lines',
                line=dict(color=line_color, width=line_width),
                opacity=opacity,
                name=None,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    def _add_frustum_fov_visualization(self, fig: go.Figure, sensor_pos: np.ndarray, 
                                     sensor_rot: np.ndarray, crop_config: LiDARSimulationCrop) -> None:
        """Add frustum FOV visualization to the figure.
        
        Args:
            fig: Plotly figure to add to
            sensor_pos: Sensor position [3]
            sensor_rot: Sensor rotation matrix [3x3]
            crop_config: LiDAR configuration object
        """
        # Convert total FOV angles to half-angles for symmetric ranges
        horizontal_fov, vertical_fov = crop_config.fov
        h_fov_half = horizontal_fov / 2
        h_fov_min = np.radians(-h_fov_half)
        h_fov_max = np.radians(h_fov_half)
        
        v_fov_half = vertical_fov / 2
        v_fov_min = np.radians(-v_fov_half)
        v_fov_max = np.radians(v_fov_half)
        
        frustum_length = 8.0
        
        # Calculate the four corners of the frustum at the far end
        corners = []
        for v_angle in [v_fov_min, v_fov_max]:
            for h_angle in [h_fov_min, h_fov_max]:
                # Point in sensor coordinates (forward = +X)
                x = frustum_length * np.cos(v_angle) * np.cos(h_angle)
                y = frustum_length * np.cos(v_angle) * np.sin(h_angle)
                z = frustum_length * np.sin(v_angle)
                
                # Transform to world coordinates
                local_point = np.array([x, y, z])
                world_point = sensor_rot @ local_point + sensor_pos
                corners.append(world_point)
        
        # corners[0]: bottom-left, corners[1]: bottom-right
        # corners[2]: top-left, corners[3]: top-right
        
        # Add lines from sensor to each corner (frustum edges)
        edge_colors = ['orange', 'darkorange', 'red', 'darkred']
        for i, corner in enumerate(corners):
            fig.add_trace(go.Scatter3d(
                x=[sensor_pos[0], corner[0]],
                y=[sensor_pos[1], corner[1]],
                z=[sensor_pos[2], corner[2]],
                mode='lines',
                line=dict(color=edge_colors[i], width=3),
                name='FOV Edge' if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo='skip'
            ))
        
        # Add rectangular outline at the far end of frustum
        # Bottom edge: bottom-left to bottom-right
        fig.add_trace(go.Scatter3d(
            x=[corners[0][0], corners[1][0]],
            y=[corners[0][1], corners[1][1]],
            z=[corners[0][2], corners[1][2]],
            mode='lines',
            line=dict(color='orange', width=2),
            name=None, showlegend=False, hoverinfo='skip'
        ))
        
        # Top edge: top-left to top-right
        fig.add_trace(go.Scatter3d(
            x=[corners[2][0], corners[3][0]],
            y=[corners[2][1], corners[3][1]],
            z=[corners[2][2], corners[3][2]],
            mode='lines',
            line=dict(color='orange', width=2),
            name=None, showlegend=False, hoverinfo='skip'
        ))
        
        # Left edge: bottom-left to top-left
        fig.add_trace(go.Scatter3d(
            x=[corners[0][0], corners[2][0]],
            y=[corners[0][1], corners[2][1]],
            z=[corners[0][2], corners[2][2]],
            mode='lines',
            line=dict(color='orange', width=2),
            name=None, showlegend=False, hoverinfo='skip'
        ))
        
        # Right edge: bottom-right to top-right
        fig.add_trace(go.Scatter3d(
            x=[corners[1][0], corners[3][0]],
            y=[corners[1][1], corners[3][1]],
            z=[corners[1][2], corners[3][2]],
            mode='lines',
            line=dict(color='orange', width=2),
            name=None, showlegend=False, hoverinfo='skip'
        ))
        
        # Add center cross at far end for better depth perception
        center = np.mean(corners, axis=0)
        fig.add_trace(go.Scatter3d(
            x=[corners[0][0], corners[3][0], None, corners[1][0], corners[2][0]],
            y=[corners[0][1], corners[3][1], None, corners[1][1], corners[2][1]],
            z=[corners[0][2], corners[3][2], None, corners[1][2], corners[2][2]],
            mode='lines',
            line=dict(color='orange', width=1, dash='dot'),
            name=None, showlegend=False, hoverinfo='skip'
        ))
    
    def get_available_options(self) -> Dict[str, List[Dict[str, str]]]:
        """Get available options for dropdowns.
        
        Returns:
            Dictionary with options for point clouds and crop types
        """
        return {
            'point_clouds': [
                {'label': 'Cube', 'value': 'cube'},
                {'label': 'Sphere', 'value': 'sphere'},
                {'label': 'Scene', 'value': 'scene'}
            ],
            'crop_types': [
                {'label': 'Range Only', 'value': 'range_only'},
                {'label': 'FOV Only', 'value': 'fov_only'},
                {'label': 'Occlusion Only', 'value': 'occlusion_only'}
            ]
        }
    
    def get_default_camera_pose(self) -> Dict[str, float]:
        """Get default camera pose parameters.
        
        Returns:
            Dictionary with default camera pose values
        """
        return {
            'azimuth': 0.0,      # Starting from +X axis
            'elevation': 0.0,    # At horizon level
            'distance': 10.0,    # 10 units from origin
            'yaw': 0.0,         # No additional yaw
            'pitch': 0.0,       # No additional pitch 
            'roll': 0.0         # No additional roll
        }
    
    def get_default_crop_params(self) -> Dict[str, Dict[str, Any]]:
        """Get default crop parameters for each crop type.
        
        Returns:
            Dictionary with default crop parameters by type
        """
        return {
            'range_only': {
                'range_max': 6.0
            },
            'fov_only': {
                'h_fov': 80.0,
                'v_fov': 40.0,
                'fov_mode': 'ellipsoid'
            },
            'occlusion_only': {}
        }
