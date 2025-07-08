"""Camera pose generation for point cloud benchmarking."""

from typing import List
import torch
import numpy as np

from .data_types import CameraPose


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
        # Calculate point cloud center and proper spatial bounds
        pc_center = point_cloud.mean(dim=0).cpu().numpy()
        
        # Calculate bounding box dimensions in each axis
        pc_min = point_cloud.min(dim=0)[0].cpu().numpy()
        pc_max = point_cloud.max(dim=0)[0].cpu().numpy()
        pc_extents = pc_max - pc_min
        
        # Use the maximum extent as the characteristic size
        pc_size = np.max(pc_extents)
        
        # Calculate diagonal distance for better scale reference
        diagonal_size = np.sqrt(np.sum(pc_extents**2))
        
        # Define distance groups based on diagonal size for proper 3D scaling
        distance_groups = {
            'close': diagonal_size * 0.75,   # Close viewing - can see details
            'medium': diagonal_size * 2.5,   # Medium distance - balanced view
            'far': diagonal_size * 6.0       # Far viewing - should trigger LOD
        }
        
        # Debug info (can be removed later)
        print(f"  Point cloud spatial analysis:")
        print(f"    Extents: {pc_extents}")
        print(f"    Max extent: {pc_size:.2f}")
        print(f"    Diagonal: {diagonal_size:.2f}")
        print(f"    Distance groups: close={distance_groups['close']:.2f}, medium={distance_groups['medium']:.2f}, far={distance_groups['far']:.2f}")
        
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
                # Use proper spatial extents for realistic offset scaling
                side_offset = self.rng.uniform(-diagonal_size * 0.2, diagonal_size * 0.2, 3)
                side_offset[2] *= 0.3  # Moderate vertical offset
                
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