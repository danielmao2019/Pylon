from typing import Dict, Union, Tuple
import torch
import numpy as np
from data.transforms.vision_3d.lidar_simulation_crop.base_fov_crop import BaseFOVCrop


class PerspectiveFOVCrop(BaseFOVCrop):
    """Perspective field-of-view cropping for camera frustum simulation.
    
    Uses perspective projection to create rectangular frustum-shaped cropping regions
    that match camera image boundaries exactly. This enables pixel-wise correspondence
    between cropped point clouds and RGB images taken from the same camera pose.
    
    Key characteristics:
    - Creates rectangular/pyramidal frustum coverage areas
    - Uses perspective projection with depth clipping planes
    - Ensures pixel-wise correspondence with camera images
    - Angular constraints applied in projected image space
    """

    def __init__(
        self,
        fov: Tuple[Union[int, float], Union[int, float]] = (90.0, 60.0),
        near_clip: float = 0.1,
        far_clip: float = 1000.0
    ):
        """Initialize perspective FOV crop.
        
        Args:
            fov: Tuple of (horizontal_fov, vertical_fov) in degrees
                - horizontal_fov: Horizontal field of view total angle 
                - vertical_fov: Vertical field of view total angle
            near_clip: Near clipping plane distance (minimum depth)
            far_clip: Far clipping plane distance (maximum depth)
        """
        super().__init__(fov)
        
        assert isinstance(near_clip, (int, float)) and near_clip > 0, f"near_clip must be positive, got {near_clip}"
        assert isinstance(far_clip, (int, float)) and far_clip > near_clip, f"far_clip must be > near_clip, got {far_clip}"
        
        self.near_clip = float(near_clip)
        self.far_clip = float(far_clip)
        
        # Pre-compute tangent values for efficiency
        self.h_tan_half = np.tan(np.radians(self.horizontal_fov / 2))
        self.v_tan_half = np.tan(np.radians(self.vertical_fov / 2))

    def _validate_fov_ranges(self, horizontal_fov: float, vertical_fov: float) -> None:
        """Validate FOV ranges for perspective cropping.
        
        Args:
            horizontal_fov: Horizontal FOV in degrees
            vertical_fov: Vertical FOV in degrees
        """
        # Perspective FOV is limited to realistic camera ranges (< 180°)
        assert 0 < horizontal_fov <= 180, f"horizontal_fov must be in (0, 180], got {horizontal_fov}"
        assert 0 < vertical_fov <= 180, f"vertical_fov must be in (0, 180], got {vertical_fov}"

    def _apply_fov_constraints(self, sensor_frame_positions: torch.Tensor) -> torch.Tensor:
        """Apply perspective field-of-view constraints using camera frustum geometry.
        
        Camera coordinate system convention (matching LiDAR FOV crop):
        - X: forward (positive into the scene, depth) 
        - Y: left (positive to the left in image)
        - Z: up (positive upward in image)
        
        This matches the sensor frame convention used by SphericalFOVCrop where 
        sensor is at origin looking down +X axis.
        
        Args:
            sensor_frame_positions: Point positions in sensor coordinate frame [N, 3]
            
        Returns:
            Boolean mask [N] indicating points within camera frustum
        """
        x, y, z = sensor_frame_positions[:, 0], sensor_frame_positions[:, 1], sensor_frame_positions[:, 2]
        
        # 1. Depth clipping: Only consider points in front of camera within depth range
        # X is the forward direction (depth)
        depth_mask = (x >= self.near_clip) & (x <= self.far_clip)
        
        if depth_mask.sum() == 0:
            return torch.zeros(len(sensor_frame_positions), dtype=torch.bool, device=sensor_frame_positions.device)
        
        # 2. Perspective projection to normalized image coordinates
        # For points at depth x, the projection is:
        #   image_y = y / x  (horizontal position on image plane, Y=left maps to image horizontal)
        #   image_z = z / x  (vertical position on image plane, Z=up maps to image vertical)
        
        # Only process points that passed depth test to avoid division by zero
        valid_x = x[depth_mask]  # depth
        valid_y = y[depth_mask]  # left/right
        valid_z = z[depth_mask]  # up/down
        
        # Project to normalized image coordinates  
        image_y = valid_y / valid_x  # horizontal on image (left/right)
        image_z = valid_z / valid_x  # vertical on image (up/down)
        
        # 3. Apply rectangular frustum constraints in projected space
        # FOV defines the angular limits, which translate to tangent limits in projected space
        h_mask = torch.abs(image_y) <= self.h_tan_half  # horizontal FOV constraint
        v_mask = torch.abs(image_z) <= self.v_tan_half  # vertical FOV constraint
        
        # Combine horizontal and vertical constraints
        projection_mask = h_mask & v_mask
        
        # 4. Map results back to full point set
        full_mask = torch.zeros(len(sensor_frame_positions), dtype=torch.bool, device=sensor_frame_positions.device)
        full_mask[depth_mask] = projection_mask
        
        return full_mask