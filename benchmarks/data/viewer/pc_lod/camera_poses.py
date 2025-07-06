"""Camera pose generation for point cloud benchmarking."""

from typing import List
import torch
import numpy as np

from data_types import CameraPose


class CameraPoseSampler:
    """Generates camera poses for point cloud benchmarking."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        
    def generate_poses_for_point_cloud(self, point_cloud: torch.Tensor, 
                                     num_poses_per_distance: int = 3) -> List[CameraPose]:
        """Generate camera poses at different distances from point cloud.
        
        Args:
            point_cloud: Point cloud tensor of shape (N, 3)
            num_poses_per_distance: Number of camera poses to generate per distance group
            
        Returns:
            List of camera poses across all distance groups
        """
        # Calculate point cloud center and bounds
        pc_center = point_cloud.mean(dim=0).numpy()
        pc_bounds = (point_cloud.min().item(), point_cloud.max().item())
        pc_size = pc_bounds[1] - pc_bounds[0]
        
        # Define distance groups based on point cloud size
        distance_groups = {
            'close': pc_size * 0.5,      # Close viewing
            'medium': pc_size * 2.0,     # Medium distance  
            'far': pc_size * 5.0         # Far viewing (should trigger LOD)
        }
        
        all_poses = []
        
        for group_name, base_distance in distance_groups.items():
            for i in range(num_poses_per_distance):
                # Generate camera position facing toward the point cloud
                theta = self.rng.uniform(0, 2 * np.pi)  # Azimuth angle
                phi = self.rng.uniform(np.pi/6, np.pi/3)  # Elevation angle (avoid top-down)
                
                # Convert spherical to cartesian coordinates
                x = base_distance * np.sin(phi) * np.cos(theta)
                y = base_distance * np.sin(phi) * np.sin(theta) 
                z = base_distance * np.cos(phi)
                
                # Add some translation to the side (perpendicular to view direction)
                side_offset = self.rng.uniform(-pc_size * 0.3, pc_size * 0.3, 3)
                side_offset[2] *= 0.2  # Less vertical offset
                
                camera_pos = pc_center + np.array([x, y, z]) + side_offset
                
                # Camera always looks toward the point cloud center
                camera_state = {
                    'eye': {
                        'x': float(camera_pos[0]),
                        'y': float(camera_pos[1]),
                        'z': float(camera_pos[2])
                    },
                    'center': {
                        'x': float(pc_center[0]), 
                        'y': float(pc_center[1]),
                        'z': float(pc_center[2])
                    },
                    'up': {'x': 0, 'y': 0, 'z': 1}  # Z-up convention
                }
                
                pose = CameraPose(
                    camera_state=camera_state,
                    distance_group=group_name,
                    distance_value=base_distance,
                    pose_id=i
                )
                
                all_poses.append(pose)
        
        return all_poses