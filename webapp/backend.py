#!/usr/bin/env python3
"""
Backend module for LiDAR simulation cropping visualization.
Handles data generation, processing, and 3D plot creation.
"""

import plotly.graph_objects as go
import numpy as np
import torch
from typing import Dict, List, Tuple, Any

# Add project root to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.lidar_crop_demo import (
    create_toy_point_cloud, 
    generate_camera_poses
)
from data.transforms.vision_3d.lidar_simulation_crop import LiDARSimulationCrop


class LiDARVisualizationBackend:
    """Backend class for handling LiDAR visualization data and processing."""
    
    def __init__(self):
        """Initialize the backend."""
        self.crop_configs = self._create_crop_configs()
    
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
    
    def get_crop_config(self, crop_type: str) -> LiDARSimulationCrop:
        """Get crop configuration by type.
        
        Args:
            crop_type: Type of cropping ('range_only', 'fov_only', 'occlusion_only')
            
        Returns:
            LiDARSimulationCrop configuration object
        """
        if crop_type not in self.crop_configs:
            raise ValueError(f"Unknown crop type: {crop_type}")
        return self.crop_configs[crop_type]
    
    def process_cropping(self, cloud_name: str, crop_type: str, anchor: str) -> Dict[str, Any]:
        """Process point cloud cropping for given configuration.
        
        Args:
            cloud_name: Name of point cloud
            crop_type: Type of cropping
            anchor: Anchor name for camera pose
            
        Returns:
            Dictionary with processed data including original points, cropped points,
            sensor information, and metadata
        """
        # Get data
        original_points = self.get_point_cloud(cloud_name)
        crop_config = self.get_crop_config(crop_type)
        
        # Generate camera poses for this cloud type
        sensor_poses = generate_camera_poses(cloud_name)
        
        # Find poses for this anchor
        anchor_poses = [pose_name for pose_name in sensor_poses.keys() 
                       if pose_name.startswith(anchor)]
        
        if not anchor_poses:
            raise ValueError(f"No poses found for anchor {anchor}")
        
        # Use base pose if available, otherwise first pose
        main_pose = anchor if anchor in anchor_poses else anchor_poses[0]
        sensor_extrinsics = sensor_poses[main_pose]
        
        # Apply cropping
        pc = {'pos': original_points, 'feat': torch.ones(len(original_points), 1)}
        cropped_pc = crop_config._call_single(pc, sensor_extrinsics, generator=torch.Generator())
        cropped_points = cropped_pc['pos']
        
        # Extract sensor info
        sensor_pos = sensor_extrinsics[:3, 3].numpy()
        sensor_rot = sensor_extrinsics[:3, :3].numpy()
        
        # Calculate reduction percentage
        reduction = (1 - len(cropped_points) / len(original_points)) * 100
        
        return {
            'original_points': original_points.numpy(),
            'cropped_points': cropped_points.numpy(),
            'sensor_pos': sensor_pos,
            'sensor_rot': sensor_rot,
            'crop_config': crop_config,
            'main_pose': main_pose,
            'reduction': reduction,
            'anchor_poses': anchor_poses
        }
    
    def create_3d_scatter_plot(self, cloud_name: str, crop_type: str, anchor: str) -> go.Figure:
        """Create a 3D scatter plot for the specified configuration.
        
        Args:
            cloud_name: Name of point cloud
            crop_type: Type of cropping
            anchor: Anchor name for camera pose
            
        Returns:
            Plotly Figure object
        """
        try:
            # Process the data
            data = self.process_cropping(cloud_name, crop_type, anchor)
            
            original_np = data['original_points']
            cropped_np = data['cropped_points']
            sensor_pos = data['sensor_pos']
            sensor_rot = data['sensor_rot']
            crop_config = data['crop_config']
            main_pose = data['main_pose']
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
            
            # Set layout
            fig.update_layout(
                title=f"{cloud_name.title()} - {crop_type.replace('_', ' ').title()} - {anchor.replace('_', ' ').title()}<br>"
                      f"<sub>Pose: {main_pose} | Points: {len(original_np):,} → {len(cropped_np):,} ({reduction:.1f}% reduction)</sub>",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y", 
                    zaxis_title="Z",
                    aspectmode='cube'
                ),
                width=1000,
                height=700,
                showlegend=True
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
        """Add FOV cone visualization to the figure.
        
        Args:
            fig: Plotly figure to add to
            sensor_pos: Sensor position [3]
            sensor_rot: Sensor rotation matrix [3x3]
            crop_config: LiDAR configuration object
        """
        h_fov = np.radians(crop_config.horizontal_fov)
        v_fov_min = np.radians(crop_config.vertical_fov[0])
        v_fov_max = np.radians(crop_config.vertical_fov[1])
        
        cone_length = 8.0
        
        # Generate cone boundary lines
        h_angles = np.linspace(-h_fov/2, h_fov/2, 8)
        v_angles = [v_fov_min, v_fov_max]
        
        # Add cone edge lines
        for v_angle in v_angles:
            cone_x, cone_y, cone_z = [], [], []
            cone_x.append(sensor_pos[0])  # Start from sensor
            cone_y.append(sensor_pos[1])
            cone_z.append(sensor_pos[2])
            
            for h_angle in h_angles:
                # Point in sensor coordinates (forward = +X)
                x = cone_length * np.cos(v_angle) * np.cos(h_angle)
                y = cone_length * np.cos(v_angle) * np.sin(h_angle)
                z = cone_length * np.sin(v_angle)
                
                # Transform to world coordinates
                local_point = np.array([x, y, z])
                world_point = sensor_rot @ local_point + sensor_pos
                
                cone_x.append(world_point[0])
                cone_y.append(world_point[1])
                cone_z.append(world_point[2])
            
            # Add FOV boundary line
            fig.add_trace(go.Scatter3d(
                x=cone_x,
                y=cone_y,
                z=cone_z,
                mode='lines',
                line=dict(color='orange', width=4),
                name=f'FOV {v_angle*180/np.pi:.0f}°' if len(fig.data) < 10 else None,
                showlegend=(len(fig.data) < 10),
                hoverinfo='skip'
            ))
    
    def get_available_options(self) -> Dict[str, List[Dict[str, str]]]:
        """Get available options for dropdowns.
        
        Returns:
            Dictionary with options for point clouds, crop types, and anchors
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
            ],
            'anchors': [
                {'label': 'Positive X', 'value': 'pos_x'},
                {'label': 'Negative X', 'value': 'neg_x'},
                {'label': 'Positive Y', 'value': 'pos_y'},
                {'label': 'Negative Y', 'value': 'neg_y'},
                {'label': 'Positive Z', 'value': 'pos_z'},
                {'label': 'Negative Z', 'value': 'neg_z'}
            ]
        }