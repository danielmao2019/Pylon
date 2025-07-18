from typing import Dict, Optional, Tuple
import numpy as np
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks.point_cloud import check_point_cloud


class LiDARSimulationCrop(BaseTransform):
    """LiDAR sensor simulation crop with range, field-of-view, and occlusion.
    
    Simulates realistic LiDAR data collection from a fixed sensor pose by applying:
    1. Range filtering: Remove points beyond sensor range
    2. Field-of-view filtering: Remove points outside horizontal/vertical FOV  
    3. Occlusion simulation: Remove occluded points using ray-casting
    
    This provides physically realistic cropping that mimics actual LiDAR limitations.
    """

    def __init__(
        self,
        sensor_position: Optional[Tuple[float, float, float]] = None,
        max_range: float = 100.0,
        horizontal_fov: float = 360.0,
        vertical_fov: Tuple[float, float] = (-30.0, 10.0),
        apply_range_filter: bool = True,
        apply_fov_filter: bool = True, 
        apply_occlusion_filter: bool = False
    ):
        """Initialize LiDAR simulation crop transform.
        
        Args:
            sensor_position: Fixed sensor position (x, y, z). If None, random position is sampled.
            max_range: Maximum sensor range in meters (typical automotive: 100-200m)
            horizontal_fov: Horizontal field of view in degrees (360° for spinning, ~120° for solid-state)
            vertical_fov: Vertical FOV as (min_elevation, max_elevation) in degrees
            apply_range_filter: Whether to apply range-based filtering
            apply_fov_filter: Whether to apply field-of-view filtering
            apply_occlusion_filter: Whether to apply occlusion simulation (ray-casting)
        """
        assert isinstance(max_range, (int, float)), f"max_range must be numeric, got {type(max_range)}"
        assert max_range > 0, f"max_range must be positive, got {max_range}"
        
        assert isinstance(horizontal_fov, (int, float)), f"horizontal_fov must be numeric, got {type(horizontal_fov)}"
        assert 0 < horizontal_fov <= 360, f"horizontal_fov must be in (0, 360], got {horizontal_fov}"
        
        assert isinstance(vertical_fov, (tuple, list)), f"vertical_fov must be tuple/list, got {type(vertical_fov)}"
        assert len(vertical_fov) == 2, f"vertical_fov must have 2 elements, got {len(vertical_fov)}"
        assert vertical_fov[0] < vertical_fov[1], f"vertical_fov min must be < max, got {vertical_fov}"
        assert -90 <= vertical_fov[0] < vertical_fov[1] <= 90, f"vertical_fov must be in [-90, 90], got {vertical_fov}"
        
        assert isinstance(apply_range_filter, bool), f"apply_range_filter must be bool, got {type(apply_range_filter)}"
        assert isinstance(apply_fov_filter, bool), f"apply_fov_filter must be bool, got {type(apply_fov_filter)}"
        assert isinstance(apply_occlusion_filter, bool), f"apply_occlusion_filter must be bool, got {type(apply_occlusion_filter)}"
        
        if sensor_position is not None:
            assert len(sensor_position) == 3, f"sensor_position must have 3 elements, got {len(sensor_position)}"
            sensor_position = torch.tensor(sensor_position, dtype=torch.float32)
        
        self.sensor_position = sensor_position
        self.max_range = float(max_range)
        self.horizontal_fov = float(horizontal_fov)
        self.vertical_fov = tuple(vertical_fov)
        self.apply_range_filter = apply_range_filter
        self.apply_fov_filter = apply_fov_filter
        self.apply_occlusion_filter = apply_occlusion_filter

    def _call_single(self, pc: Dict[str, torch.Tensor], generator: torch.Generator) -> Dict[str, torch.Tensor]:
        """Apply LiDAR simulation crop to point cloud.
        
        Args:
            pc: Point cloud dictionary with 'pos' key and optional feature keys
            generator: Random number generator for reproducible results
            
        Returns:
            Cropped point cloud dictionary
        """
        check_point_cloud(pc)
        
        positions = pc['pos']  # Shape: (N, 3)
        
        # Generate or use provided sensor position
        if self.sensor_position is None:
            sensor_pos = self._sample_sensor_position(positions, generator)
        else:
            # Align sensor position to positions.device if needed
            if self.sensor_position.device != positions.device:
                self.sensor_position = self.sensor_position.to(positions.device)
            sensor_pos = self.sensor_position
        
        # Start with all points
        valid_mask = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
        
        # Apply range filter
        if self.apply_range_filter:
            range_mask = self._apply_range_filter(positions, sensor_pos)
            valid_mask = valid_mask & range_mask
        
        # Apply field-of-view filter
        if self.apply_fov_filter:
            fov_mask = self._apply_fov_filter(positions, sensor_pos)
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

    def _sample_sensor_position(self, positions: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Sample random sensor position around the point cloud.
        
        Args:
            positions: Point cloud positions [N, 3]
            generator: Random number generator
            
        Returns:
            Sensor position [3]
        """
        # Compute point cloud bounds
        min_vals = positions.min(dim=0)[0]
        max_vals = positions.max(dim=0)[0]
        pc_center = (min_vals + max_vals) / 2
        pc_extent = (max_vals - min_vals).max()
        
        # Sample sensor position outside the point cloud with some margin
        margin = pc_extent * 0.5  # 50% margin around point cloud
        
        # Sample on a sphere around the point cloud center
        # Generate random direction
        direction = torch.randn(3, generator=generator, device=positions.device)
        direction = direction / torch.norm(direction)
        
        # Place sensor at distance of pc_extent + margin from center
        sensor_distance = pc_extent / 2 + margin
        sensor_pos = pc_center + direction * sensor_distance
        
        return sensor_pos

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

    def _apply_fov_filter(self, positions: torch.Tensor, sensor_pos: torch.Tensor) -> torch.Tensor:
        """Apply field-of-view filtering.
        
        Args:
            positions: Point cloud positions [N, 3]
            sensor_pos: Sensor position [3]
            
        Returns:
            Boolean mask [N] indicating points within FOV
        """
        # Convert to sensor-centric coordinates
        relative_pos = positions - sensor_pos.unsqueeze(0)  # [N, 3]
        
        # Compute horizontal angle (azimuth) - angle in XY plane from +X axis
        azimuth = torch.atan2(relative_pos[:, 1], relative_pos[:, 0])  # [-π, π]
        azimuth_deg = torch.rad2deg(azimuth)  # [-180, 180]
        
        # Compute vertical angle (elevation) - angle from XY plane
        xy_distance = torch.norm(relative_pos[:, :2], dim=1)
        elevation = torch.atan2(relative_pos[:, 2], xy_distance)  # [-π/2, π/2]
        elevation_deg = torch.rad2deg(elevation)  # [-90, 90]
        
        # Apply horizontal FOV constraint
        if self.horizontal_fov >= 360.0:
            h_mask = torch.ones_like(azimuth_deg, dtype=torch.bool)
        else:
            # For < 360°, assume FOV is centered around +X axis (0°)
            half_h_fov = self.horizontal_fov / 2
            h_mask = torch.abs(azimuth_deg) <= half_h_fov
        
        # Apply vertical FOV constraint
        v_mask = (elevation_deg >= self.vertical_fov[0]) & (elevation_deg <= self.vertical_fov[1])
        
        return h_mask & v_mask

    def _apply_occlusion_filter(self, positions: torch.Tensor, sensor_pos: torch.Tensor, 
                               current_mask: torch.Tensor) -> torch.Tensor:
        """Apply occlusion filtering using simplified ray-casting.
        
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
        
        # Compute distances from sensor to all valid points
        distances = torch.norm(valid_positions - sensor_pos.unsqueeze(0), dim=1)
        
        # Sort points by distance (closer points can occlude farther ones)
        distance_order = torch.argsort(distances)
        sorted_positions = valid_positions[distance_order]
        sorted_distances = distances[distance_order]
        
        # Simplified occlusion: for each point, check if any closer point is "nearby" in angular space
        occlusion_mask = torch.ones(valid_positions.shape[0], dtype=torch.bool, device=positions.device)
        
        angular_threshold = 0.05  # radians (~3 degrees) - points closer than this can occlude
        
        for i in range(1, len(sorted_positions)):  # Skip first point (closest, can't be occluded)
            current_point = sorted_positions[i]
            current_distance = sorted_distances[i]
            
            # Check against all closer points
            closer_points = sorted_positions[:i]
            closer_distances = sorted_distances[:i]
            
            # Compute angular distances between current point and closer points
            # Both normalized to unit vectors from sensor
            current_dir = (current_point - sensor_pos) / current_distance
            closer_dirs = (closer_points - sensor_pos.unsqueeze(0)) / closer_distances.unsqueeze(1)
            
            # Angular distance using dot product: angle = arccos(dot(a, b))
            dot_products = torch.sum(current_dir.unsqueeze(0) * closer_dirs, dim=1)
            dot_products = torch.clamp(dot_products, -1.0, 1.0)  # Avoid numerical issues
            angular_distances = torch.acos(dot_products)
            
            # If any closer point is within angular threshold, mark as occluded
            if torch.any(angular_distances < angular_threshold):
                original_idx = distance_order[i]
                occlusion_mask[original_idx] = False
        
        # Map back to original indexing
        full_occlusion_mask = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
        full_occlusion_mask[current_mask] = occlusion_mask
        
        return full_occlusion_mask
