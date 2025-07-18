#!/usr/bin/env python3
"""
Demo script showing LiDAR simulation cropping with various sensor poses and filtering options.
Creates visualizations showing original point clouds and cropped results.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path for imports
import sys
sys.path.append('.')

from data.transforms.vision_3d.lidar_simulation_crop import LiDARSimulationCrop


def create_toy_point_cloud(cloud_type='cube', num_points=2000, noise=0.02):
    """Create synthetic point clouds for demonstration.
    
    Args:
        cloud_type: 'cube', 'sphere', or 'scene'
        num_points: Number of points to generate
        noise: Amount of noise to add
        
    Returns:
        Point cloud as torch.Tensor [N, 3]
    """
    if cloud_type == 'cube':
        # Create a cube with points on surface and interior
        edge_points = []
        # Generate points on each face
        for face in range(6):
            n_face = num_points // 6
            if face == 0:  # +X face
                x = torch.ones(n_face) * 2.0
                y = torch.rand(n_face) * 4.0 - 2.0
                z = torch.rand(n_face) * 4.0 - 2.0
            elif face == 1:  # -X face
                x = torch.ones(n_face) * -2.0
                y = torch.rand(n_face) * 4.0 - 2.0
                z = torch.rand(n_face) * 4.0 - 2.0
            elif face == 2:  # +Y face
                x = torch.rand(n_face) * 4.0 - 2.0
                y = torch.ones(n_face) * 2.0
                z = torch.rand(n_face) * 4.0 - 2.0
            elif face == 3:  # -Y face
                x = torch.rand(n_face) * 4.0 - 2.0
                y = torch.ones(n_face) * -2.0
                z = torch.rand(n_face) * 4.0 - 2.0
            elif face == 4:  # +Z face
                x = torch.rand(n_face) * 4.0 - 2.0
                y = torch.rand(n_face) * 4.0 - 2.0
                z = torch.ones(n_face) * 2.0
            else:  # -Z face
                x = torch.rand(n_face) * 4.0 - 2.0
                y = torch.rand(n_face) * 4.0 - 2.0
                z = torch.ones(n_face) * -2.0
            
            face_points = torch.stack([x, y, z], dim=1)
            edge_points.append(face_points)
        
        points = torch.cat(edge_points, dim=0)
        
    elif cloud_type == 'sphere':
        # Create a sphere
        phi = torch.rand(num_points) * 2 * np.pi  # Azimuth
        theta = torch.rand(num_points) * np.pi    # Elevation
        r = torch.rand(num_points) ** (1/3) * 2.0  # Uniform volume distribution
        
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        
        points = torch.stack([x, y, z], dim=1)
        
    elif cloud_type == 'scene':
        # Create a simple scene with multiple objects
        # Ground plane
        ground_x = torch.rand(num_points // 4) * 20.0 - 10.0
        ground_y = torch.rand(num_points // 4) * 20.0 - 10.0
        ground_z = torch.zeros(num_points // 4) - 2.0
        
        # Building 1
        build1_x = torch.rand(num_points // 4) * 3.0 + 2.0
        build1_y = torch.rand(num_points // 4) * 3.0 - 1.5
        build1_z = torch.rand(num_points // 4) * 5.0 - 2.0
        
        # Building 2
        build2_x = torch.rand(num_points // 4) * 2.0 - 6.0
        build2_y = torch.rand(num_points // 4) * 4.0 + 1.0
        build2_z = torch.rand(num_points // 4) * 3.0 - 2.0
        
        # Tree-like structure
        tree_x = torch.rand(num_points // 4) * 1.0 - 0.5
        tree_y = torch.rand(num_points // 4) * 1.0 + 6.0
        tree_z = torch.rand(num_points // 4) * 4.0 - 2.0
        
        all_x = torch.cat([ground_x, build1_x, build2_x, tree_x])
        all_y = torch.cat([ground_y, build1_y, build2_y, tree_y])
        all_z = torch.cat([ground_z, build1_z, build2_z, tree_z])
        
        points = torch.stack([all_x, all_y, all_z], dim=1)
    
    # Add noise
    if noise > 0:
        points = points + torch.randn_like(points) * noise
    
    return points


def create_sensor_poses():
    """Create various sensor poses for demonstration.
    
    Returns:
        Dict of pose name to 4x4 extrinsics matrix
    """
    poses = {}
    
    # Pose 1: Sensor at origin looking down +X axis
    pose1 = torch.eye(4)
    pose1[:3, 3] = torch.tensor([0.0, 0.0, 0.0])
    poses['origin_forward'] = pose1
    
    # Pose 2: Sensor elevated looking down
    pose2 = torch.eye(4)
    pose2[:3, 3] = torch.tensor([0.0, 0.0, 8.0])
    # Rotate to look down (-Z direction)
    pose2[0, 0] = 0.0; pose2[0, 2] = -1.0
    pose2[2, 0] = 1.0; pose2[2, 2] = 0.0
    poses['elevated_down'] = pose2
    
    # Pose 3: Sensor to the side looking toward center
    pose3 = torch.eye(4)
    pose3[:3, 3] = torch.tensor([8.0, 0.0, 2.0])
    # Rotate to look toward -X direction
    pose3[0, 0] = -1.0
    pose3[1, 1] = -1.0
    poses['side_view'] = pose3
    
    # Pose 4: Sensor at an angle
    pose4 = torch.eye(4)
    pose4[:3, 3] = torch.tensor([5.0, 5.0, 3.0])
    # Rotate to look toward origin (approximately)
    # This is a simplified rotation - in practice you'd use proper rotation matrices
    angle = np.pi / 4
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    pose4[0, 0] = cos_a; pose4[0, 1] = sin_a
    pose4[1, 0] = -sin_a; pose4[1, 1] = cos_a
    poses['angled_view'] = pose4
    
    return poses


def plot_point_cloud_comparison(original_points, cropped_results, sensor_poses, save_dir='lidar_demo_plots'):
    """Create comparison plots showing original vs cropped point clouds.
    
    Args:
        original_points: Original point cloud [N, 3]
        cropped_results: Dict of pose_name -> cropped_points
        sensor_poses: Dict of pose_name -> 4x4 matrix
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy for plotting
    orig_np = original_points.numpy()
    
    for pose_name, cropped_points in cropped_results.items():
        cropped_np = cropped_points.numpy()
        sensor_pos = sensor_poses[pose_name][:3, 3].numpy()
        
        fig = plt.figure(figsize=(15, 5))
        
        # Get sensor orientation for all plots
        sensor_rotation = sensor_poses[pose_name][:3, :3].numpy()
        forward_dir = sensor_rotation[:, 0]  # Forward direction in world frame
        arrow_length = 2.0
        
        # Calculate common axis limits based on original point cloud and sensor position
        all_points = np.vstack([orig_np, sensor_pos.reshape(1, -1)])
        margin = 2.0  # Add some margin around the data
        x_min, x_max = all_points[:, 0].min() - margin, all_points[:, 0].max() + margin
        y_min, y_max = all_points[:, 1].min() - margin, all_points[:, 1].max() + margin
        z_min, z_max = all_points[:, 2].min() - margin, all_points[:, 2].max() + margin
        
        # Original point cloud
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(orig_np[:, 0], orig_np[:, 1], orig_np[:, 2], 
                   c='blue', alpha=0.6, s=1, label='Original')
        ax1.scatter([sensor_pos[0]], [sensor_pos[1]], [sensor_pos[2]], 
                   c='red', s=100, marker='o', label='Sensor')
        ax1.quiver(sensor_pos[0], sensor_pos[1], sensor_pos[2],
                  forward_dir[0], forward_dir[1], forward_dir[2],
                  length=arrow_length, color='orange', arrow_length_ratio=0.3, linewidth=2)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_zlim(z_min, z_max)
        ax1.set_title('Original Point Cloud')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        ax1.legend()
        
        # Cropped point cloud
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(cropped_np[:, 0], cropped_np[:, 1], cropped_np[:, 2], 
                   c='green', alpha=0.8, s=1, label='Cropped')
        ax2.scatter([sensor_pos[0]], [sensor_pos[1]], [sensor_pos[2]], 
                   c='red', s=100, marker='o', label='Sensor')
        ax2.quiver(sensor_pos[0], sensor_pos[1], sensor_pos[2],
                  forward_dir[0], forward_dir[1], forward_dir[2],
                  length=arrow_length, color='orange', arrow_length_ratio=0.3, linewidth=2)
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_zlim(z_min, z_max)
        ax2.set_title(f'After LiDAR Crop ({len(cropped_np)}/{len(orig_np)} points)')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
        ax2.legend()
        
        # Overlay comparison with ultra-high-contrast colors
        ax3 = fig.add_subplot(133, projection='3d')
        
        # Show removed points in very dark color with high opacity
        ax3.scatter(orig_np[:, 0], orig_np[:, 1], orig_np[:, 2], 
                   c='black', alpha=0.8, s=3, label='Removed', edgecolors='none')
        # Show kept points in very bright color with black outlines
        ax3.scatter(cropped_np[:, 0], cropped_np[:, 1], cropped_np[:, 2], 
                   c='yellow', alpha=1.0, s=8, label='Kept', edgecolors='red', linewidths=0.5)
        # Show sensor very prominently with orientation arrow
        ax3.scatter([sensor_pos[0]], [sensor_pos[1]], [sensor_pos[2]], 
                   c='magenta', s=200, marker='o', label='Sensor', edgecolors='white', linewidths=2)
        
        # Draw arrow from sensor position in facing direction
        ax3.quiver(sensor_pos[0], sensor_pos[1], sensor_pos[2],
                  forward_dir[0], forward_dir[1], forward_dir[2],
                  length=arrow_length, color='cyan', arrow_length_ratio=0.3, 
                  linewidth=3, label='Facing Direction')
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.set_zlim(z_min, z_max)
        ax3.set_title('Overlay Comparison')
        ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/lidar_crop_{pose_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_dir}/lidar_crop_{pose_name}.png")
        print(f"  Original points: {len(orig_np)}, Cropped points: {len(cropped_np)}, "
              f"Reduction: {100*(1-len(cropped_np)/len(orig_np)):.1f}%")


def main():
    """Run the LiDAR cropping demonstration."""
    print("LiDAR Simulation Crop Demonstration")
    print("=" * 40)
    
    # Create toy point clouds
    point_clouds = {
        'cube': create_toy_point_cloud('cube', 3000),
        'sphere': create_toy_point_cloud('sphere', 2000),
        'scene': create_toy_point_cloud('scene', 4000)
    }
    
    # Create sensor poses
    sensor_poses = create_sensor_poses()
    
    # Create different LiDAR configurations
    lidar_configs = {
        'range_only': LiDARSimulationCrop(
            max_range=6.0,
            apply_range_filter=True,
            apply_fov_filter=False,
            apply_occlusion_filter=False
        ),
        'range_and_fov': LiDARSimulationCrop(
            max_range=10.0,
            horizontal_fov=120.0,
            vertical_fov=(-30.0, 30.0),
            apply_range_filter=True,
            apply_fov_filter=True,
            apply_occlusion_filter=False
        ),
        'full_simulation': LiDARSimulationCrop(
            max_range=8.0,
            horizontal_fov=90.0,
            vertical_fov=(-20.0, 20.0),
            apply_range_filter=True,
            apply_fov_filter=True,
            apply_occlusion_filter=True
        )
    }
    
    # Test each configuration
    for cloud_name, points in point_clouds.items():
        print(f"\nTesting with {cloud_name} point cloud ({len(points)} points)")
        
        for config_name, lidar_crop in lidar_configs.items():
            print(f"\n  Configuration: {config_name}")
            
            # Create point cloud dictionary
            pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
            
            # Test with different sensor poses
            cropped_results = {}
            for pose_name, sensor_extrinsics in sensor_poses.items():
                try:
                    cropped_pc = lidar_crop._call_single(pc, sensor_extrinsics, generator=torch.Generator())
                    cropped_results[pose_name] = cropped_pc['pos']
                    print(f"    {pose_name}: {len(points)} -> {len(cropped_pc['pos'])} points")
                except Exception as e:
                    print(f"    {pose_name}: ERROR - {str(e)}")
                    continue
            
            # Create plots
            if cropped_results:
                save_dir = f'lidar_demo_plots/{cloud_name}_{config_name}'
                plot_point_cloud_comparison(points, cropped_results, sensor_poses, save_dir)
    
    print(f"\nDemo complete! Check the 'lidar_demo_plots/' directory for visualizations.")


if __name__ == "__main__":
    main()