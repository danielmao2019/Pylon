from typing import Dict, Tuple
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks.point_cloud import check_point_cloud


class FOVFilter(BaseTransform):
    """Field-of-view filtering for LiDAR sensor simulation.
    
    Filters points based on horizontal and vertical field-of-view constraints,
    simulating the angular coverage limitations of real LiDAR sensors.
    """

    def __init__(
        self,
        horizontal_fov: float = 360.0,
        vertical_fov: Tuple[float, float] = (-30.0, 10.0)
    ):
        """Initialize FOV filter.
        
        Args:
            horizontal_fov: Horizontal field of view in degrees (360° for spinning, ~120° for solid-state)
            vertical_fov: Vertical FOV as (min_elevation, max_elevation) in degrees
        """
        assert isinstance(horizontal_fov, (int, float)), f"horizontal_fov must be numeric, got {type(horizontal_fov)}"
        assert 0 < horizontal_fov <= 360, f"horizontal_fov must be in (0, 360], got {horizontal_fov}"
        
        assert isinstance(vertical_fov, (tuple, list)), f"vertical_fov must be tuple/list, got {type(vertical_fov)}"
        assert len(vertical_fov) == 2, f"vertical_fov must have 2 elements, got {len(vertical_fov)}"
        assert vertical_fov[0] < vertical_fov[1], f"vertical_fov min must be < max, got {vertical_fov}"
        assert -90 <= vertical_fov[0] < vertical_fov[1] <= 90, f"vertical_fov must be in [-90, 90], got {vertical_fov}"
        
        self.horizontal_fov = float(horizontal_fov)
        self.vertical_fov = tuple(vertical_fov)

    def _call_single(self, pc: Dict[str, torch.Tensor], sensor_extrinsics: torch.Tensor,
                    *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Apply FOV filtering to point cloud.
        
        Args:
            pc: Point cloud dictionary with 'pos' key and optional feature keys
            sensor_extrinsics: 4x4 sensor pose matrix (sensor-to-world transform)
            
        Returns:
            Filtered point cloud dictionary
        """
        check_point_cloud(pc)
        
        assert isinstance(sensor_extrinsics, torch.Tensor), f"sensor_extrinsics must be torch.Tensor, got {type(sensor_extrinsics)}"
        assert sensor_extrinsics.shape == (4, 4), f"sensor_extrinsics must be 4x4, got {sensor_extrinsics.shape}"
        
        positions = pc['pos']  # Shape: [N, 3]
        
        # Align sensor extrinsics to positions.device if needed
        if sensor_extrinsics.device != positions.device:
            sensor_extrinsics = sensor_extrinsics.to(positions.device)
        
        # Transform points to sensor coordinate frame for FOV calculations
        # Convert positions to homogeneous coordinates
        positions_homo = torch.cat([positions, torch.ones(positions.shape[0], 1, device=positions.device)], dim=1)
        
        # Transform to sensor frame: inverse(sensor_extrinsics) @ world_points
        # sensor_extrinsics is sensor-to-world, so we need world-to-sensor (inverse)
        world_to_sensor = torch.inverse(sensor_extrinsics)
        sensor_frame_positions = (world_to_sensor @ positions_homo.T).T[:, :3]
        
        # Apply FOV constraints
        fov_mask = self._apply_fov_constraints(sensor_frame_positions)
        
        # Apply mask to all keys in point cloud
        filtered_pc = {}
        for key, tensor in pc.items():
            if key == 'pos':
                filtered_pc[key] = positions[fov_mask]
            else:
                # Assume features have same first dimension as positions
                filtered_pc[key] = tensor[fov_mask]
        
        return filtered_pc

    def _apply_fov_constraints(self, sensor_frame_positions: torch.Tensor) -> torch.Tensor:
        """Apply field-of-view constraints to points in sensor coordinate frame.
        
        Args:
            sensor_frame_positions: Point positions in sensor coordinate frame [N, 3]
            
        Returns:
            Boolean mask [N] indicating points within FOV
        """
        # Points are in sensor frame, so sensor is at origin looking down +X axis
        # X: forward, Y: left, Z: up in sensor frame
        
        # Compute horizontal angle (azimuth) - angle in XY plane from +X axis
        azimuth = torch.atan2(sensor_frame_positions[:, 1], sensor_frame_positions[:, 0])  # [-π, π]
        azimuth_deg = torch.rad2deg(azimuth)  # [-180, 180]
        
        # Compute vertical angle (elevation) - angle from XY plane
        xy_distance = torch.norm(sensor_frame_positions[:, :2], dim=1)
        elevation = torch.atan2(sensor_frame_positions[:, 2], xy_distance)  # [-π/2, π/2]
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