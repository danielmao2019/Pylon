#!/usr/bin/env python3
"""
Backend module for LiDAR simulation cropping visualization.
Handles data generation, processing, and 3D plot creation.
"""

import plotly.graph_objects as go
import numpy as np
import torch
from typing import Dict, List, Tuple, Any

from demos.data.transforms.vision_3d.lidar_simulation_crop.lidar_crop_demo import create_toy_point_cloud
from data.transforms.vision_3d.lidar_simulation_crop import LiDARSimulationCrop


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
                apply_range_filter=True,
                apply_fov_filter=False,
                apply_occlusion_filter=False
            ),
            'fov_only': LiDARSimulationCrop(
                max_range=100.0,  # Very large range so no range filtering
                horizontal_fov=80.0,
                vertical_fov=(-20.0, 20.0),
                apply_range_filter=False,
                apply_fov_filter=True,
                apply_occlusion_filter=False
            ),
            'occlusion_only': LiDARSimulationCrop(
                max_range=100.0,  # Very large range so no range filtering
                horizontal_fov=360.0,  # Full circle so no FOV filtering
                vertical_fov=(-90.0, 90.0),  # Full sphere so no FOV filtering
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
        
        Args:
            azimuth: Horizontal angle from +X axis in degrees [0, 360]
            elevation: Vertical angle from horizon in degrees [-90, 90]
            distance: Distance from origin [1, 20]
            yaw: Camera yaw rotation in degrees [-180, 180] (relative to look-at-origin)
            pitch: Camera pitch rotation in degrees [-90, 90]
            roll: Camera roll rotation in degrees [-180, 180]
            
        Returns:
            4x4 extrinsics matrix (sensor-to-world transformation)
        """
        # Convert to radians
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        
        # Convert spherical to cartesian coordinates
        x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = distance * np.sin(elevation_rad)
        
        eye = torch.tensor([x, y, z], dtype=torch.float32)
        target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        
        # Create look-at matrix (camera looking toward origin)
        forward = target - eye
        forward = forward / torch.norm(forward)
        
        # Handle special case when forward is parallel to up vector
        cos_angle = torch.dot(forward, up)
        if torch.abs(cos_angle) > 0.99:  # Nearly parallel
            if torch.abs(forward[0]) < 0.9:
                up = torch.tensor([1.0, 0.0, 0.0])
            else:
                up = torch.tensor([0.0, 1.0, 0.0])
        
        # Compute right vector
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        
        # Recompute up vector to ensure orthogonality
        up = torch.cross(right, forward)
        up = up / torch.norm(up)
        
        # Create base rotation matrix (world-to-sensor)
        # In sensor frame: +X forward, +Y left, +Z up
        rotation = torch.stack([forward, -right, up], dim=0)  # 3x3
        
        # Apply additional rotations (yaw, pitch, roll) in sensor frame
        # Yaw (around Z)
        cy, sy = torch.cos(torch.tensor(yaw_rad)), torch.sin(torch.tensor(yaw_rad))
        R_yaw = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=torch.float32)
        
        # Pitch (around Y)
        cp, sp = torch.cos(torch.tensor(pitch_rad)), torch.sin(torch.tensor(pitch_rad))
        R_pitch = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=torch.float32)
        
        # Roll (around X)
        cr, sr = torch.cos(torch.tensor(roll_rad)), torch.sin(torch.tensor(roll_rad))
        R_roll = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=torch.float32)
        
        # Apply rotations in order: yaw, then pitch, then roll
        final_rotation = rotation @ R_yaw @ R_pitch @ R_roll
        
        # Create 4x4 extrinsics matrix (sensor-to-world)
        extrinsics = torch.eye(4)
        extrinsics[:3, :3] = final_rotation.T  # Transpose for sensor-to-world
        extrinsics[:3, 3] = eye
        
        return extrinsics

    def get_crop_config(self, crop_type: str, **params) -> LiDARSimulationCrop:
        """Get crop configuration by type with dynamic parameters.
        
        Args:
            crop_type: Type of cropping ('range_only', 'fov_only', 'occlusion_only')
            **params: Dynamic parameters for crop configuration
                - range_max: Maximum range for range_only cropping
                - h_fov: Horizontal FOV for fov_only cropping
                - v_fov_span: Vertical FOV span for fov_only cropping
            
        Returns:
            LiDARSimulationCrop configuration object
        """
        if crop_type == 'range_only':
            range_max = params.get('range_max', 6.0)
            return LiDARSimulationCrop(
                max_range=range_max,
                apply_range_filter=True,
                apply_fov_filter=False,
                apply_occlusion_filter=False
            )
        elif crop_type == 'fov_only':
            h_fov = params.get('h_fov', 80.0)
            v_fov_span = params.get('v_fov_span', 40.0)  # Total span around center (0)
            v_fov_min = -v_fov_span / 2
            v_fov_max = v_fov_span / 2
            return LiDARSimulationCrop(
                max_range=100.0,  # Very large range so no range filtering
                horizontal_fov=h_fov,
                vertical_fov=(v_fov_min, v_fov_max),
                apply_range_filter=False,
                apply_fov_filter=True,
                apply_occlusion_filter=False
            )
        elif crop_type == 'occlusion_only':
            return LiDARSimulationCrop(
                max_range=100.0,  # Very large range so no range filtering
                horizontal_fov=360.0,  # Full circle so no FOV filtering
                vertical_fov=(-90.0, 90.0),  # Full sphere so no FOV filtering
                apply_range_filter=False,
                apply_fov_filter=False,
                apply_occlusion_filter=True
            )
        else:
            raise ValueError(f"Unknown crop type: {crop_type}")
    
    def process_cropping(self, cloud_name: str, crop_type: str, 
                        azimuth: float, elevation: float, distance: float,
                        yaw: float, pitch: float, roll: float, **crop_params) -> Dict[str, Any]:
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
            **crop_params: Dynamic crop parameters (range_max, h_fov, v_fov_span)
            
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
        pc = {'pos': original_points, 'feat': torch.ones(len(original_points), 1)}
        cropped_pc = crop_config._call_single(pc, sensor_extrinsics, generator=torch.Generator())
        cropped_points = cropped_pc['pos']
        
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
                              yaw: float, pitch: float, roll: float, **crop_params) -> go.Figure:
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
            **crop_params: Dynamic crop parameters
            
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
                self._add_fov_visualization(fig, sensor_pos, sensor_rot, crop_config)
            
            # Set layout with fixed axis ranges for consistent scaling
            fig.update_layout(
                title=f"{cloud_name.title()} - {crop_type.replace('_', ' ').title()}<br>"
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
                             sensor_rot: np.ndarray, crop_config: LiDARSimulationCrop) -> None:
        """Add FOV frustum visualization to the figure.
        
        Args:
            fig: Plotly figure to add to
            sensor_pos: Sensor position [3]
            sensor_rot: Sensor rotation matrix [3x3]
            crop_config: LiDAR configuration object
        """
        h_fov = np.radians(crop_config.horizontal_fov)
        v_fov_min = np.radians(crop_config.vertical_fov[0])
        v_fov_max = np.radians(crop_config.vertical_fov[1])
        
        frustum_length = 8.0
        
        # Calculate the four corners of the frustum at the far end
        corners = []
        for v_angle in [v_fov_min, v_fov_max]:
            for h_angle in [-h_fov/2, h_fov/2]:
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
    
    def get_default_crop_params(self) -> Dict[str, Dict[str, float]]:
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
                'v_fov_span': 40.0
            },
            'occlusion_only': {}
        }