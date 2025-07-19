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


def create_toy_point_cloud(cloud_type='cube', num_points=2000, noise=0.02, seed=42):
    """Create synthetic point clouds for demonstration.
    
    Args:
        cloud_type: 'cube', 'sphere', or 'scene'
        num_points: Number of points to generate
        noise: Amount of noise to add
        seed: Random seed for reproducible generation
        
    Returns:
        Point cloud as torch.Tensor [N, 3]
    """
    # Set seed for reproducible point cloud generation
    torch.manual_seed(seed)
    np.random.seed(seed)
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


def create_look_at_matrix(eye, target, up=None):
    """Create a look-at transformation matrix.
    
    Args:
        eye: Camera position [3]
        target: Target position to look at [3] 
        up: Up vector [3], defaults to [0, 0, 1]
        
    Returns:
        4x4 extrinsics matrix (sensor-to-world transformation)
    """
    if up is None:
        up = torch.tensor([0.0, 0.0, 1.0])
    
    # Convert to float tensors
    eye = eye.float()
    target = target.float()  
    up = up.float()
    
    # Compute view direction (forward vector)
    forward = target - eye
    forward = forward / torch.norm(forward)
    
    # Handle special case when forward is parallel to up vector
    cos_angle = torch.dot(forward, up)
    if torch.abs(cos_angle) > 0.99:  # Nearly parallel
        # Choose different up vector
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
    
    # Create rotation matrix (world-to-sensor)
    # In sensor frame: +X forward, +Y left, +Z up
    rotation = torch.stack([forward, -right, up], dim=0)  # 3x3
    
    # Create 4x4 extrinsics matrix (sensor-to-world)
    extrinsics = torch.eye(4)
    extrinsics[:3, :3] = rotation.T  # Transpose for sensor-to-world
    extrinsics[:3, 3] = eye
    
    return extrinsics


def create_sensor_poses():
    """Create systematic sensor poses based on 6 anchor axes with variations.
    
    Starts from 6 anchor poses along +/-X, +/-Y, +/-Z axes, all facing toward origin,
    then creates variations by adjusting rotation and position slightly.
    
    Returns:
        Dict of pose name to 4x4 extrinsics matrix
    """
    poses = {}
    origin = torch.tensor([0.0, 0.0, 0.0])
    base_distance = 6.0
    
    # 6 anchor poses along main axes, all looking toward origin
    anchor_positions = {
        'pos_x': torch.tensor([base_distance, 0.0, 0.0]),
        'neg_x': torch.tensor([-base_distance, 0.0, 0.0]), 
        'pos_y': torch.tensor([0.0, base_distance, 0.0]),
        'neg_y': torch.tensor([0.0, -base_distance, 0.0]),
        'pos_z': torch.tensor([0.0, 0.0, base_distance]),
        'neg_z': torch.tensor([0.0, 0.0, -base_distance])
    }
    
    # Create anchor poses
    for name, position in anchor_positions.items():
        pose = create_look_at_matrix(position, origin)
        poses[f'anchor_{name}'] = pose
    
    # Create variations by adjusting position and rotation
    variations = [
        # Closer positions with slight offsets
        ('close_pos_x', torch.tensor([3.5, 0.5, 0.8])),
        ('close_neg_y', torch.tensor([0.3, -4.0, 1.2])),
        ('close_pos_z', torch.tensor([-0.8, 0.4, 3.8])),
        
        # Farther positions with angular offsets  
        ('far_angled_x', torch.tensor([8.5, 2.0, 1.5])),
        ('far_angled_y', torch.tensor([1.8, -9.0, 2.5])),
        ('far_elevated', torch.tensor([1.2, 1.5, 9.5])),
        
        # Mixed diagonal positions
        ('diagonal_1', torch.tensor([4.5, 3.8, 2.2])),
        ('diagonal_2', torch.tensor([-3.2, 4.1, -2.8])),
        ('diagonal_3', torch.tensor([2.8, -2.5, 4.5])),
        
        # Asymmetric positions for varied coverage
        ('asym_1', torch.tensor([7.2, -1.8, 0.5])),
        ('asym_2', torch.tensor([-2.1, 5.5, 3.1])),
        ('asym_3', torch.tensor([0.8, 2.2, -5.8]))
    ]
    
    # Create variation poses, all looking toward origin
    for name, position in variations:
        pose = create_look_at_matrix(position, origin)
        poses[name] = pose
    
    return poses


def plot_point_cloud_comparison(original_points, cropped_results, sensor_poses, save_dir='lidar_demo_plots', config_name=None, lidar_config=None):
    """Create grid of overlaid plots showing original vs cropped point clouds for selected sensor poses.
    
    Args:
        original_points: Original point cloud [N, 3]
        cropped_results: Dict of pose_name -> cropped_points
        sensor_poses: Dict of pose_name -> 4x4 matrix
        save_dir: Directory to save plots
        config_name: Configuration name for visualization enhancements
        lidar_config: LiDAR configuration object for parameter visualization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy for plotting
    orig_np = original_points.numpy()
    
    # Select interesting poses for visualization - prioritize diversity and meaningful results
    priority_poses = [
        # Anchor poses from main axes
        'anchor_pos_x', 'anchor_neg_x', 'anchor_pos_y', 
        'anchor_neg_y', 'anchor_pos_z', 'anchor_neg_z',
        # Close variations with good coverage
        'close_pos_x', 'close_neg_y', 'close_pos_z',
        # Diagonal positions for varied perspectives  
        'diagonal_1', 'diagonal_2', 'diagonal_3',
        # Asymmetric positions
        'asym_1', 'asym_2', 'asym_3'
    ]
    
    # Filter to only available poses and select subset based on point retention
    available_poses = set(cropped_results.keys())
    candidate_poses = [p for p in priority_poses if p in available_poses]
    
    # Calculate point retention rates to select most interesting poses
    pose_stats = []
    for pose_name in candidate_poses:
        if pose_name in cropped_results:
            original_count = len(original_points)
            kept_count = len(cropped_results[pose_name])
            retention_rate = kept_count / original_count if original_count > 0 else 0
            pose_stats.append((pose_name, retention_rate, kept_count))
    
    # Sort by diversity criteria: prefer poses with varied retention rates (avoid all 0% or all 100%)
    pose_stats.sort(key=lambda x: abs(x[1] - 0.5))  # Prefer poses near 50% retention for interesting results
    
    # Select up to 12 most interesting poses
    max_poses = 12
    selected_poses = [stat[0] for stat in pose_stats[:max_poses]]
    
    # Ensure we have anchor poses for systematic coverage
    anchor_poses = [p for p in selected_poses if p.startswith('anchor_')]
    if len(anchor_poses) < 6:  # Add missing anchor poses
        missing_anchors = [p for p in priority_poses[:6] if p in available_poses and p not in selected_poses]
        selected_poses.extend(missing_anchors[:6-len(anchor_poses)])
    
    # Limit to reasonable number for visualization
    selected_poses = selected_poses[:max_poses]
    num_poses = len(selected_poses)
    
    # Determine grid size
    if num_poses <= 6:
        rows, cols = 2, 3
        figsize = (20, 12)
    elif num_poses <= 9:
        rows, cols = 3, 3
        figsize = (20, 16)
    else:
        rows, cols = 3, 4
        figsize = (24, 16)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, subplot_kw={'projection': '3d'})
    axes = axes.flatten() if num_poses > 1 else [axes]
    
    # Create pose titles for display
    pose_titles = {}
    for pose_name in selected_poses:
        if pose_name.startswith('anchor_'):
            axis = pose_name.replace('anchor_', '').replace('_', ' ').title()
            pose_titles[pose_name] = f'Anchor {axis}'
        elif pose_name.startswith('close_'):
            axis = pose_name.replace('close_', '').replace('_', ' ').title()
            pose_titles[pose_name] = f'Close {axis}'
        elif pose_name.startswith('far_'):
            axis = pose_name.replace('far_', '').replace('_', ' ').title()
            pose_titles[pose_name] = f'Far {axis}'
        elif pose_name.startswith('diagonal_'):
            num = pose_name.replace('diagonal_', '')
            pose_titles[pose_name] = f'Diagonal {num}'
        elif pose_name.startswith('asym_'):
            num = pose_name.replace('asym_', '')
            pose_titles[pose_name] = f'Asymmetric {num}'
        else:
            pose_titles[pose_name] = pose_name.replace('_', ' ').title()
    
    # Calculate common axis limits for all subplots
    all_sensor_positions = np.array([sensor_poses[pose][:3, 3].numpy() for pose in selected_poses])
    all_points = np.vstack([orig_np, all_sensor_positions])
    margin = 2.0
    x_min, x_max = all_points[:, 0].min() - margin, all_points[:, 0].max() + margin
    y_min, y_max = all_points[:, 1].min() - margin, all_points[:, 1].max() + margin
    z_min, z_max = all_points[:, 2].min() - margin, all_points[:, 2].max() + margin
    
    # Create subplot for each sensor pose
    for i, pose_name in enumerate(selected_poses):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if pose_name not in cropped_results:
            ax.set_title(f'{pose_titles.get(pose_name, pose_name)}\nNo data')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            continue
            
        cropped_points = cropped_results[pose_name]
        cropped_np = cropped_points.numpy()
        sensor_pos = sensor_poses[pose_name][:3, 3].numpy()
        sensor_rotation = sensor_poses[pose_name][:3, :3].numpy()
        forward_dir = sensor_rotation[:, 0]
        arrow_length = 2.0
        
        # Create mask for removed vs kept points
        # For visualization, we'll show all original points and highlight kept ones
        
        # Show all original points in blue (removed points) with low opacity
        ax.scatter(orig_np[:, 0], orig_np[:, 1], orig_np[:, 2], 
                  c='blue', alpha=0.3, s=2, label='Removed')
        
        # Show kept points in red (overlaid on top) with high opacity
        ax.scatter(cropped_np[:, 0], cropped_np[:, 1], cropped_np[:, 2], 
                  c='red', alpha=1.0, s=6, label='Kept')
        
        # Show sensor position and orientation
        ax.scatter([sensor_pos[0]], [sensor_pos[1]], [sensor_pos[2]], 
                  c='black', s=150, marker='o', label='Sensor', edgecolors='white', linewidths=2)
        
        ax.quiver(sensor_pos[0], sensor_pos[1], sensor_pos[2],
                 forward_dir[0], forward_dir[1], forward_dir[2],
                 length=arrow_length, color='green', arrow_length_ratio=0.3, 
                 linewidth=3, alpha=0.8)
        
        # Add visualization enhancements based on config type
        if config_name and lidar_config:
            # Add range visualization if range filter is enabled
            if lidar_config.apply_range_filter:
                _add_range_visualization(ax, sensor_pos, lidar_config.max_range)
            # Add FOV visualization if FOV filter is enabled
            if lidar_config.apply_fov_filter:
                _add_fov_visualization(ax, sensor_pos, sensor_rotation, lidar_config)
        
        # Calculate equal axis range to force square aspect ratio
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2
        center_z = (z_max + z_min) / 2
        
        # Set equal limits around center
        half_range = max_range / 2
        ax.set_xlim(center_x - half_range, center_x + half_range)
        ax.set_ylim(center_y - half_range, center_y + half_range)
        ax.set_zlim(center_z - half_range, center_z + half_range)
        
        # Force equal aspect ratio to prevent stretching
        ax.set_box_aspect([1,1,1])
        
        ax.set_title(f'{pose_titles[pose_name]}\n{len(cropped_np)}/{len(orig_np)} pts ({100*(1-len(cropped_np)/len(orig_np)):.1f}% reduced)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Only show legend on first subplot
        if i == 0:
            ax.legend()
    
    # Hide unused subplots
    for j in range(len(selected_poses), len(axes)):
        if j < len(axes):
            axes[j].set_visible(False)
    
    plt.suptitle(f'{config_name.replace("_", " ").title()} Configuration', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lidar_crop_all_views.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    total_reductions = []
    for pose_name in selected_poses:
        if pose_name in cropped_results:
            cropped_np = cropped_results[pose_name].numpy()
            reduction = 100*(1-len(cropped_np)/len(orig_np))
            total_reductions.append(reduction)
            print(f"  {pose_titles.get(pose_name, pose_name)}: {len(orig_np)} -> {len(cropped_np)} points ({reduction:.1f}% reduction)")
    
    if total_reductions:
        avg_reduction = sum(total_reductions) / len(total_reductions)
        print(f"  Average reduction: {avg_reduction:.1f}%")
    
    print(f"Saved: {save_dir}/lidar_crop_all_views.png")


def _add_range_visualization(ax, sensor_pos, max_range):
    """Add range sphere visualization to plot.
    
    Args:
        ax: 3D matplotlib axis
        sensor_pos: Sensor position [3]
        max_range: Maximum range value
    """
    # Create a wireframe sphere to show range limit
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = max_range * np.outer(np.cos(u), np.sin(v)) + sensor_pos[0]
    sphere_y = max_range * np.outer(np.sin(u), np.sin(v)) + sensor_pos[1]
    sphere_z = max_range * np.outer(np.ones(np.size(u)), np.cos(v)) + sensor_pos[2]
    
    # Plot wireframe sphere in semi-transparent purple
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.3, color='purple', linewidth=1)
    

def _add_fov_visualization(ax, sensor_pos, sensor_rotation, lidar_config):
    """Add FOV cone visualization to plot.
    
    Args:
        ax: 3D matplotlib axis
        sensor_pos: Sensor position [3]
        sensor_rotation: Sensor rotation matrix [3x3]
        lidar_config: LiDAR configuration object
    """
    # Get FOV parameters
    h_fov = np.radians(lidar_config.horizontal_fov)
    v_fov_min = np.radians(lidar_config.vertical_fov[0])
    v_fov_max = np.radians(lidar_config.vertical_fov[1])
    
    # Create cone visualization
    cone_length = 5.0  # Visual cone length
    
    # Generate cone points in sensor coordinate system
    h_angles = np.linspace(-h_fov/2, h_fov/2, 10)
    v_angles = np.array([v_fov_min, v_fov_max])
    
    cone_points = []
    # Add apex (sensor position)
    cone_points.append([0, 0, 0])
    
    # Generate cone edges
    for v_angle in v_angles:
        for h_angle in h_angles:
            # Point in sensor coordinates (forward = +X)
            x = cone_length * np.cos(v_angle) * np.cos(h_angle)
            y = cone_length * np.cos(v_angle) * np.sin(h_angle)
            z = cone_length * np.sin(v_angle)
            cone_points.append([x, y, z])
    
    cone_points = np.array(cone_points)
    
    # Transform to world coordinates
    # cone_points are in sensor frame, transform to world frame
    world_points = (sensor_rotation @ cone_points.T).T + sensor_pos
    
    # Draw cone edges from apex to boundary
    apex = world_points[0]
    boundary_points = world_points[1:]
    
    # Draw lines from apex to boundary points
    for point in boundary_points[::2]:  # Skip some for cleaner visualization
        ax.plot([apex[0], point[0]], [apex[1], point[1]], [apex[2], point[2]], 
                color='orange', alpha=0.6, linewidth=1.5)
    
    # Draw boundary rectangle
    # Bottom edge
    bottom_left = boundary_points[0]
    bottom_right = boundary_points[len(h_angles)-1]
    ax.plot([bottom_left[0], bottom_right[0]], [bottom_left[1], bottom_right[1]], 
            [bottom_left[2], bottom_right[2]], color='orange', alpha=0.8, linewidth=2)
    
    # Top edge
    top_left = boundary_points[len(h_angles)]
    top_right = boundary_points[-1]
    ax.plot([top_left[0], top_right[0]], [top_left[1], top_right[1]], 
            [top_left[2], top_right[2]], color='orange', alpha=0.8, linewidth=2)
    
    # Side edges
    ax.plot([bottom_left[0], top_left[0]], [bottom_left[1], top_left[1]], 
            [bottom_left[2], top_left[2]], color='orange', alpha=0.8, linewidth=2)
    ax.plot([bottom_right[0], top_right[0]], [bottom_right[1], top_right[1]], 
            [bottom_right[2], top_right[2]], color='orange', alpha=0.8, linewidth=2)


def main():
    """Run the LiDAR cropping demonstration."""
    print("LiDAR Simulation Crop Demonstration")
    print("=" * 40)
    
    # Create toy point clouds with fixed seed for reproducibility
    point_clouds = {
        'cube': create_toy_point_cloud('cube', 3000, seed=42),
        'sphere': create_toy_point_cloud('sphere', 2000, seed=43),
        'scene': create_toy_point_cloud('scene', 4000, seed=44)
    }
    
    # Create sensor poses
    sensor_poses = create_sensor_poses()
    
    # Create different LiDAR configurations - reordered: single filters -> mixed -> full
    lidar_configs = {
        # Single filter types
        'range_only': LiDARSimulationCrop(
            max_range=6.0,
            apply_range_filter=True,
            apply_fov_filter=False,
            apply_occlusion_filter=False
        ),
        'fov_only': LiDARSimulationCrop(
            max_range=100.0,  # Very large range so no range filtering
            horizontal_fov=120.0,
            vertical_fov=(-30.0, 30.0),
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
        ),
        # Mixed filter types
        'range_and_fov': LiDARSimulationCrop(
            max_range=10.0,
            horizontal_fov=120.0,
            vertical_fov=(-30.0, 30.0),
            apply_range_filter=True,
            apply_fov_filter=True,
            apply_occlusion_filter=False
        ),
        # Full simulation
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
                plot_point_cloud_comparison(points, cropped_results, sensor_poses, save_dir, config_name, lidar_crop)
    
    print(f"\nDemo complete! Check the 'lidar_demo_plots/' directory for visualizations.")


if __name__ == "__main__":
    main()
