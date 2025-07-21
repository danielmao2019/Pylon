from typing import Dict, Union
import torch
import numpy as np
from data.transforms.base_transform import BaseTransform
from utils.input_checks.point_cloud import check_point_cloud


class LiDARSimulationCrop(BaseTransform):
    """LiDAR sensor simulation crop with range, field-of-view, and occlusion.
    
    Simulates realistic LiDAR data collection from a fixed sensor pose by applying:
    1. Range filtering: Remove points beyond sensor range
    2. Field-of-view filtering: Remove points outside horizontal/vertical FOV  
    3. Occlusion simulation: Remove occluded points using ray-casting
    
    This provides physically realistic cropping that mimics actual LiDAR limitations.
    Takes a 4x4 extrinsics matrix defining the sensor pose (position + rotation).
    """

    def __init__(
        self,
        max_range: float = 100.0,
        horizontal_fov: Union[int, float] = 360.0,
        vertical_fov: Union[int, float] = 40.0,
        apply_range_filter: bool = True,
        apply_fov_filter: bool = True, 
        apply_occlusion_filter: bool = False
    ):
        """Initialize LiDAR simulation crop transform.
        
        Args:
            max_range: Maximum sensor range in meters (typical automotive: 100-200m)
            horizontal_fov: Horizontal field of view total angle in degrees (360° for spinning, ~120° for solid-state)
            vertical_fov: Vertical field of view total angle in degrees (e.g., 40° means [-20°, +20°])
            apply_range_filter: Whether to apply range-based filtering
            apply_fov_filter: Whether to apply field-of-view filtering
            apply_occlusion_filter: Whether to apply occlusion simulation (ray-casting)
        """
        assert isinstance(max_range, (int, float)), f"max_range must be numeric, got {type(max_range)}"
        assert max_range > 0, f"max_range must be positive, got {max_range}"
        
        assert isinstance(horizontal_fov, (int, float)), f"horizontal_fov must be numeric, got {type(horizontal_fov)}"
        assert 0 < horizontal_fov <= 360, f"horizontal_fov must be in (0, 360], got {horizontal_fov}"
        
        assert isinstance(vertical_fov, (int, float)), f"vertical_fov must be numeric, got {type(vertical_fov)}"
        assert 0 < vertical_fov <= 180, f"vertical_fov must be in (0, 180], got {vertical_fov}"
        
        assert isinstance(apply_range_filter, bool), f"apply_range_filter must be bool, got {type(apply_range_filter)}"
        assert isinstance(apply_fov_filter, bool), f"apply_fov_filter must be bool, got {type(apply_fov_filter)}"
        assert isinstance(apply_occlusion_filter, bool), f"apply_occlusion_filter must be bool, got {type(apply_occlusion_filter)}"
        
        self.max_range = float(max_range)
        self.horizontal_fov = float(horizontal_fov)
        self.vertical_fov = float(vertical_fov)
        self.apply_range_filter = apply_range_filter
        self.apply_fov_filter = apply_fov_filter
        self.apply_occlusion_filter = apply_occlusion_filter

    def _call_single(self, pc: Dict[str, torch.Tensor], sensor_extrinsics: torch.Tensor, *args, generator: torch.Generator) -> Dict[str, torch.Tensor]:
        """Apply LiDAR simulation crop to point cloud.
        
        Args:
            pc: Point cloud dictionary with 'pos' key and optional feature keys
            sensor_extrinsics: 4x4 sensor pose matrix (world-to-sensor transform)
            generator: Random number generator (not used, kept for API compatibility)
            
        Returns:
            Cropped point cloud dictionary
        """
        check_point_cloud(pc)
        
        # Validate sensor extrinsics matrix
        assert isinstance(sensor_extrinsics, torch.Tensor), f"sensor_extrinsics must be torch.Tensor, got {type(sensor_extrinsics)}"
        assert sensor_extrinsics.shape == (4, 4), f"sensor_extrinsics must be 4x4, got {sensor_extrinsics.shape}"
        
        positions = pc['pos']  # Shape: (N, 3)
        
        # Align sensor extrinsics to positions.device if needed
        if sensor_extrinsics.device != positions.device:
            sensor_extrinsics = sensor_extrinsics.to(positions.device)
        
        # Extract sensor position from extrinsics matrix
        sensor_pos = sensor_extrinsics[:3, 3]  # Translation component
        
        # Transform points to sensor coordinate frame for FOV calculations
        if self.apply_fov_filter:
            # Convert positions to homogeneous coordinates
            positions_homo = torch.cat([positions, torch.ones(positions.shape[0], 1, device=positions.device)], dim=1)
            # Transform to sensor frame: inverse(sensor_extrinsics) @ world_points
            # sensor_extrinsics is sensor-to-world, so we need world-to-sensor (inverse)
            world_to_sensor = torch.inverse(sensor_extrinsics)
            sensor_frame_positions = (world_to_sensor @ positions_homo.T).T[:, :3]
        else:
            sensor_frame_positions = None
        
        # Start with all points
        valid_mask = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
        
        # Apply range filter
        if self.apply_range_filter:
            range_mask = self._apply_range_filter(positions, sensor_pos)
            valid_mask = valid_mask & range_mask
        
        # Apply field-of-view filter
        if self.apply_fov_filter:
            fov_mask = self._apply_fov_filter(sensor_frame_positions)
            valid_mask = valid_mask & fov_mask
        
        # Apply occlusion filter (most expensive, do last)
        if self.apply_occlusion_filter:
            occlusion_mask = self._apply_occlusion_filter(positions, sensor_pos, valid_mask)
            valid_mask = valid_mask & occlusion_mask
        
        # Apply final mask to all keys in point cloud
        cropped_pc = {}
        for key, tensor in pc.items():
            if key == 'pos':
                cropped_pc[key] = positions[valid_mask]
            else:
                # Assume features have same first dimension as positions
                cropped_pc[key] = tensor[valid_mask]
        
        return cropped_pc

    def _apply_range_filter(self, positions: torch.Tensor, sensor_pos: torch.Tensor) -> torch.Tensor:
        """Apply range-based filtering.
        
        Args:
            positions: Point cloud positions [N, 3]
            sensor_pos: Sensor position [3]
            
        Returns:
            Boolean mask [N] indicating points within range
        """
        distances = torch.norm(positions - sensor_pos.unsqueeze(0), dim=1)
        return distances <= self.max_range

    def _apply_fov_filter(self, sensor_frame_positions: torch.Tensor) -> torch.Tensor:
        """Apply field-of-view filtering using points in sensor coordinate frame.
        
        Args:
            sensor_frame_positions: Point positions in sensor coordinate frame [N, 3]
            
        Returns:
            Boolean mask [N] indicating points within FOV
        """
        # Compute angles in sensor frame (sensor at origin looking down +X axis)
        # X: forward, Y: left, Z: up
        x, y, z = sensor_frame_positions[:, 0], sensor_frame_positions[:, 1], sensor_frame_positions[:, 2]
        
        # Horizontal angle (azimuth) - angle in XY plane from +X axis [-180°, 180°]
        azimuth_deg = torch.rad2deg(torch.atan2(y, x))
        
        # Vertical angle (elevation) - angle from XY plane [-90°, 90°]
        xy_distance = torch.norm(sensor_frame_positions[:, :2], dim=1)
        elevation_deg = torch.rad2deg(torch.atan2(z, xy_distance))
        
        # Apply horizontal FOV constraint
        half_h_fov = self.horizontal_fov / 2
        h_mask = torch.abs(azimuth_deg) <= half_h_fov
        
        # Apply vertical FOV constraint (symmetric around 0°)
        half_v_fov = self.vertical_fov / 2
        v_mask = torch.abs(elevation_deg) <= half_v_fov
        
        return h_mask & v_mask

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
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            # Fallback: use simple distance calculation (slower but works)
            return self._fallback_density_analysis(positions)
        
        tree = cKDTree(positions_np)
        
        # Find k-nearest neighbors for each point
        distances, indices = tree.query(positions_np, k=k_neighbors+1)  # +1 because includes self
        
        # Remove self-distance (first column is always 0)
        neighbor_distances = distances[:, 1:]
        
        # Calculate average distance to k-nearest neighbors for each point
        avg_distances = np.mean(neighbor_distances, axis=1)
        
        # Use 90th percentile as robust density measure (less sensitive to outliers)
        return float(np.percentile(avg_distances, 90))
    
    def _fallback_density_analysis(self, positions: torch.Tensor) -> float:
        """Fallback density analysis when scipy is not available."""
        # Sample subset of points for efficiency
        n_sample = min(1000, positions.shape[0])
        sample_indices = torch.randperm(positions.shape[0])[:n_sample]
        sample_positions = positions[sample_indices]
        
        # Calculate pairwise distances for sample
        distances = torch.cdist(sample_positions, sample_positions)
        
        # For each point, find distance to 10th closest point (excluding self)
        k = min(10, n_sample - 1)
        sorted_distances, _ = torch.sort(distances, dim=1)
        kth_distances = sorted_distances[:, k]  # k+1 because of self at index 0
        
        # Return 90th percentile
        return float(torch.quantile(kth_distances, 0.9))
    
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
    
    def _apply_occlusion_filter(self, positions: torch.Tensor, sensor_pos: torch.Tensor, 
                               current_mask: torch.Tensor) -> torch.Tensor:
        """Apply occlusion filtering using ray-based voxel approach.
        
        Args:
            positions: Point cloud positions [N, 3]
            sensor_pos: Sensor position [3]
            current_mask: Current valid points mask [N]
            
        Returns:
            Boolean mask [N] indicating non-occluded points
        """
        # Only consider currently valid points for occlusion testing
        valid_positions = positions[current_mask]
        
        if valid_positions.shape[0] == 0:
            return current_mask
        
        # Automatically determine optimal voxel size based on point density
        voxel_size = self._determine_voxel_size(valid_positions)
        
        # Convert to numpy for processing
        valid_positions_np = valid_positions.cpu().numpy()
        sensor_pos_np = sensor_pos.cpu().numpy()
        
        # Compute distances and sort by distance (closer points first)
        distances = np.linalg.norm(valid_positions_np - sensor_pos_np, axis=1)
        distance_order = np.argsort(distances)
        sorted_positions = valid_positions_np[distance_order]
        sorted_distances = distances[distance_order]
        
        # Create voxel grid bounds
        all_positions = np.vstack([valid_positions_np, sensor_pos_np[np.newaxis, :]])
        min_coords = all_positions.min(axis=0) - voxel_size
        max_coords = all_positions.max(axis=0) + voxel_size
        
        def pos_to_voxel_key(pos):
            """Convert position to voxel key for hashing."""
            voxel_coords = np.floor((pos - min_coords) / voxel_size).astype(int)
            return tuple(voxel_coords)
        
        # Use dictionary for fast voxel occupancy lookup
        occupied_voxels = {}
        occlusion_mask = np.ones(len(valid_positions_np), dtype=bool)
        
        # Ray density factor: check 80% of ray length for occlusion
        ray_density_factor = 0.8
        
        # Process points in distance order (closer points first)
        for i, (point, distance) in enumerate(zip(sorted_positions, sorted_distances)):
            ray_direction = (point - sensor_pos_np) / distance
            
            # Check ray for occlusion - sample based on voxel size and distance
            max_check_distance = distance * ray_density_factor
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
        
        # Map back to original indexing
        full_occlusion_mask = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
        full_occlusion_mask[current_mask] = torch.from_numpy(occlusion_mask).to(positions.device)
        
        return full_occlusion_mask
