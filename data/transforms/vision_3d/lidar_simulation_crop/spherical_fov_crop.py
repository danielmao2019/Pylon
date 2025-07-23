from typing import Dict, Union, Tuple
import torch
from data.transforms.vision_3d.lidar_simulation_crop.base_fov_crop import BaseFOVCrop


class SphericalFOVCrop(BaseFOVCrop):
    """Spherical field-of-view cropping for LiDAR sensor simulation.
    
    Uses spherical coordinates (azimuth/elevation angles) to create cone-shaped
    cropping regions that simulate real LiDAR sensor angular coverage limitations.
    This is the traditional LiDAR FOV cropping mode.
    
    Key characteristics:
    - Creates cone-shaped coverage areas
    - Uses azimuth (horizontal) and elevation (vertical) angles  
    - Suitable for spinning LiDAR sensors (up to 360° horizontal FOV)
    - Angular constraints applied in spherical coordinate space
    """

    def _validate_fov_ranges(self, horizontal_fov: float, vertical_fov: float) -> None:
        """Validate FOV ranges for spherical cropping.
        
        Args:
            horizontal_fov: Horizontal FOV in degrees
            vertical_fov: Vertical FOV in degrees
        """
        # Spherical FOV supports full 360° horizontal coverage for spinning LiDAR
        assert 0 < horizontal_fov <= 360, f"horizontal_fov must be in (0, 360], got {horizontal_fov}"
        assert 0 < vertical_fov <= 180, f"vertical_fov must be in (0, 180], got {vertical_fov}"

    def _apply_fov_constraints(self, sensor_frame_positions: torch.Tensor) -> torch.Tensor:
        """Apply spherical field-of-view constraints to points in sensor coordinate frame.
        
        Uses spherical coordinates to define cone-shaped FOV regions typical of LiDAR sensors.
        
        Args:
            sensor_frame_positions: Point positions in sensor coordinate frame [N, 3]
            
        Returns:
            Boolean mask [N] indicating points within spherical FOV
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
        
        # Apply vertical FOV constraint (symmetric around 0°)
        half_v_fov = self.vertical_fov / 2
        v_mask = torch.abs(elevation_deg) <= half_v_fov
        
        return h_mask & v_mask
