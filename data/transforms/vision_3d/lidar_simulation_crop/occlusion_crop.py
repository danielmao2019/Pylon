from typing import Dict
import torch
import numpy as np
from scipy.spatial import cKDTree
from data.transforms.base_transform import BaseTransform
from utils.input_checks.point_cloud import check_point_cloud


class OcclusionCrop(BaseTransform):
    """Occlusion cropping for LiDAR sensor simulation.
    
    Simulates realistic occlusion effects using ray-based voxel approach.
    Automatically determines optimal voxel size based on point cloud density.
    """

    def __init__(self, ray_density_factor: float = 0.8):
        """Initialize occlusion crop.
        
        Args:
            ray_density_factor: Fraction of ray length to check for occlusion (0.8 = check 80%)
        """
        assert isinstance(ray_density_factor, (int, float)), f"ray_density_factor must be numeric, got {type(ray_density_factor)}"
        assert 0.1 <= ray_density_factor <= 1.0, f"ray_density_factor must be in [0.1, 1.0], got {ray_density_factor}"
        
        self.ray_density_factor = float(ray_density_factor)

    def _call_single(self, pc: Dict[str, torch.Tensor], sensor_pos: torch.Tensor,
                    *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Apply occlusion cropping to point cloud.
        
        Args:
            pc: Point cloud dictionary with 'pos' key and optional feature keys
            sensor_pos: Sensor position as [3] tensor
            
        Returns:
            Cropped point cloud dictionary
        """
        check_point_cloud(pc)
        
        assert isinstance(sensor_pos, torch.Tensor), f"sensor_pos must be torch.Tensor, got {type(sensor_pos)}"
        assert sensor_pos.shape == (3,), f"sensor_pos must be [3], got {sensor_pos.shape}"
        
        positions = pc['pos']  # Shape: [N, 3]
        
        if positions.shape[0] == 0:
            return pc
        
        # Align sensor_pos to positions.device if needed
        if sensor_pos.device != positions.device:
            sensor_pos = sensor_pos.to(positions.device)
        
        # Apply ray-based occlusion cropping
        occlusion_mask = self._apply_ray_based_occlusion(positions, sensor_pos)
        
        # Apply mask to all keys in point cloud
        cropped_pc = {}
        for key, tensor in pc.items():
            if key == 'pos':
                cropped_pc[key] = positions[occlusion_mask]
            else:
                # Assume features have same first dimension as positions
                cropped_pc[key] = tensor[occlusion_mask]
        
        return cropped_pc

    def _analyze_point_density(self, positions: torch.Tensor, k_neighbors: int = 10) -> float:
        """Analyze point cloud density to determine optimal voxel size.
        
        Args:
            positions: Point cloud positions [N, 3]
            k_neighbors: Number of neighbors to consider for density estimation
            
        Returns:
            90th percentile of average neighbor distances (robust density measure)
        """
        positions_np = positions.cpu().numpy()
        
        # Build KD-tree for efficient neighbor search
        tree = cKDTree(positions_np)
        
        # Find k-nearest neighbors for each point
        distances, _ = tree.query(positions_np, k=k_neighbors+1)  # +1 because includes self
        
        # Remove self-distance (first column is always 0)
        neighbor_distances = distances[:, 1:]
        
        # Calculate average distance to k-nearest neighbors for each point
        avg_distances = np.mean(neighbor_distances, axis=1)
        
        # Use 90th percentile as robust density measure (less sensitive to outliers)
        return float(np.percentile(avg_distances, 90))

    def _determine_voxel_size(self, positions: torch.Tensor) -> float:
        """Automatically determine optimal voxel size based on point density.
        
        Args:
            positions: Point cloud positions [N, 3]
            
        Returns:
            Optimal voxel size for ray-based occlusion filtering
        """
        if positions.shape[0] < 10:
            return 0.2  # Default for very small point clouds
        
        # Analyze point density
        density_spacing = self._analyze_point_density(positions)
        
        # Base voxel size: 3x the 90th percentile spacing
        # This groups nearby points into surfaces while preserving detail
        base_voxel_size = density_spacing * 3.0
        
        # Clamp to reasonable bounds
        min_voxel_size = 0.05  # 5cm minimum for precision
        max_voxel_size = 2.0   # 2m maximum for efficiency
        
        voxel_size = max(min_voxel_size, min(base_voxel_size, max_voxel_size))
        
        return voxel_size

    def _apply_ray_based_occlusion(self, positions: torch.Tensor, sensor_pos: torch.Tensor) -> torch.Tensor:
        """Apply occlusion cropping using ray-based voxel approach.
        
        Args:
            positions: Point cloud positions [N, 3]
            sensor_pos: Sensor position [3]
            
        Returns:
            Boolean mask [N] indicating non-occluded points
        """
        # Automatically determine optimal voxel size based on point density
        voxel_size = self._determine_voxel_size(positions)
        
        # Convert to numpy for processing
        positions_np = positions.cpu().numpy()
        sensor_pos_np = sensor_pos.cpu().numpy()
        
        # Compute distances and sort by distance (closer points first)
        distances = np.linalg.norm(positions_np - sensor_pos_np, axis=1)
        distance_order = np.argsort(distances)
        sorted_positions = positions_np[distance_order]
        sorted_distances = distances[distance_order]
        
        # Create voxel grid bounds
        all_positions = np.vstack([positions_np, sensor_pos_np[np.newaxis, :]])
        min_coords = all_positions.min(axis=0) - voxel_size
        
        def pos_to_voxel_key(pos):
            """Convert position to voxel key for hashing."""
            voxel_coords = np.floor((pos - min_coords) / voxel_size).astype(int)
            return tuple(voxel_coords)
        
        # Use dictionary for fast voxel occupancy lookup
        occupied_voxels = {}
        occlusion_mask = np.ones(len(positions_np), dtype=bool)
        
        # Process points in distance order (closer points first)
        for i, (point, distance) in enumerate(zip(sorted_positions, sorted_distances)):
            ray_direction = (point - sensor_pos_np) / distance
            
            # Check ray for occlusion - sample based on voxel size and distance
            max_check_distance = distance * self.ray_density_factor
            num_samples = max(10, int(max_check_distance / voxel_size))
            
            is_occluded = False
            for sample_idx in range(1, num_samples + 1):  # Skip sensor position
                sample_distance = max_check_distance * (sample_idx / num_samples)
                sample_point = sensor_pos_np + ray_direction * sample_distance
                voxel_key = pos_to_voxel_key(sample_point)
                
                if voxel_key in occupied_voxels:
                    is_occluded = True
                    break
            
            original_idx = distance_order[i]
            if is_occluded:
                occlusion_mask[original_idx] = False
            else:
                # Mark point's voxel as occupied for future occlusion checks
                point_voxel_key = pos_to_voxel_key(point)
                occupied_voxels[point_voxel_key] = i
        
        return torch.from_numpy(occlusion_mask).to(positions.device)