from typing import Dict, Union, Tuple
import torch
import numpy as np
from data.transforms.base_transform import BaseTransform
from utils.input_checks.point_cloud import check_point_cloud


class CameraFrustumCrop(BaseTransform):
    """Camera frustum cropping for pixel-wise correspondence with RGB images.
    
    Unlike LiDAR FOV cropping which uses spherical coordinates (creating cones),
    camera frustum cropping uses perspective projection to create rectangular
    frustums that match camera image boundaries exactly.
    
    This ensures perfect pixel-wise correspondence between cropped point clouds
    and RGB images taken from the same camera pose.
    """

    def __init__(
        self,
        fov: Tuple[Union[int, float], Union[int, float]] = (90.0, 60.0),
        near_clip: float = 0.1,
        far_clip: float = 1000.0
    ):
        """Initialize camera frustum crop.
        
        Args:
            fov: Tuple of (horizontal_fov, vertical_fov) in degrees
                - horizontal_fov: Horizontal field of view total angle 
                - vertical_fov: Vertical field of view total angle
            near_clip: Near clipping plane distance (minimum depth)
            far_clip: Far clipping plane distance (maximum depth)
        """
        assert isinstance(fov, tuple), f"fov must be tuple, got {type(fov)}"
        assert len(fov) == 2, f"fov must be tuple of length 2, got length {len(fov)}"
        horizontal_fov, vertical_fov = fov
        
        assert isinstance(horizontal_fov, (int, float)), f"horizontal_fov must be numeric, got {type(horizontal_fov)}"
        assert 0 < horizontal_fov <= 180, f"horizontal_fov must be in (0, 180], got {horizontal_fov}"
        
        assert isinstance(vertical_fov, (int, float)), f"vertical_fov must be numeric, got {type(vertical_fov)}"
        assert 0 < vertical_fov <= 180, f"vertical_fov must be in (0, 180], got {vertical_fov}"
        
        assert isinstance(near_clip, (int, float)) and near_clip > 0, f"near_clip must be positive, got {near_clip}"
        assert isinstance(far_clip, (int, float)) and far_clip > near_clip, f"far_clip must be > near_clip, got {far_clip}"
        
        self.horizontal_fov = float(horizontal_fov)
        self.vertical_fov = float(vertical_fov)
        self.near_clip = float(near_clip)
        self.far_clip = float(far_clip)
        
        # Pre-compute tangent values for efficiency
        self.h_tan_half = np.tan(np.radians(horizontal_fov / 2))
        self.v_tan_half = np.tan(np.radians(vertical_fov / 2))

    def _call_single(self, pc: Dict[str, torch.Tensor], sensor_extrinsics: torch.Tensor,
                    *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Apply camera frustum cropping to point cloud.
        
        Args:
            pc: Point cloud dictionary with 'pos' key and optional feature keys
            sensor_extrinsics: 4x4 sensor pose matrix (sensor-to-world transform)
            
        Returns:
            Cropped point cloud dictionary with points inside camera frustum
        """
        check_point_cloud(pc)
        
        assert isinstance(sensor_extrinsics, torch.Tensor), f"sensor_extrinsics must be torch.Tensor, got {type(sensor_extrinsics)}"
        assert sensor_extrinsics.shape == (4, 4), f"sensor_extrinsics must be 4x4, got {sensor_extrinsics.shape}"
        
        positions = pc['pos']  # Shape: [N, 3]
        
        # Align sensor extrinsics to positions.device if needed
        if sensor_extrinsics.device != positions.device:
            sensor_extrinsics = sensor_extrinsics.to(positions.device)
        
        # Transform points to camera coordinate frame for frustum calculations
        # Convert positions to homogeneous coordinates
        positions_homo = torch.cat([positions, torch.ones(positions.shape[0], 1, device=positions.device)], dim=1)
        
        # Transform to camera frame: inverse(sensor_extrinsics) @ world_points
        # sensor_extrinsics is sensor-to-world, so we need world-to-sensor (inverse)
        world_to_camera = torch.inverse(sensor_extrinsics)
        camera_frame_positions = (world_to_camera @ positions_homo.T).T[:, :3]
        
        # Apply camera frustum constraints
        frustum_mask = self._apply_camera_frustum_constraints(camera_frame_positions)
        
        # Apply mask to all keys in point cloud
        cropped_pc = {}
        for key, tensor in pc.items():
            if key == 'pos':
                cropped_pc[key] = positions[frustum_mask]
            else:
                # Assume features have same first dimension as positions
                cropped_pc[key] = tensor[frustum_mask]
        
        return cropped_pc

    def _apply_camera_frustum_constraints(self, camera_frame_positions: torch.Tensor) -> torch.Tensor:
        """Apply camera frustum constraints using perspective projection.
        
        Camera coordinate system convention (matching LiDAR FOV crop):
        - X: forward (positive into the scene, depth) 
        - Y: left (positive to the left in image)
        - Z: up (positive upward in image)
        
        This matches the sensor frame convention used by FOVCrop where 
        sensor is at origin looking down +X axis.
        
        Args:
            camera_frame_positions: Point positions in camera coordinate frame [N, 3]
            
        Returns:
            Boolean mask [N] indicating points within camera frustum
        """
        x, y, z = camera_frame_positions[:, 0], camera_frame_positions[:, 1], camera_frame_positions[:, 2]
        
        # 1. Depth clipping: Only consider points in front of camera within depth range
        # X is the forward direction (depth)
        depth_mask = (x >= self.near_clip) & (x <= self.far_clip)
        
        if depth_mask.sum() == 0:
            return torch.zeros(len(camera_frame_positions), dtype=torch.bool, device=camera_frame_positions.device)
        
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
        full_mask = torch.zeros(len(camera_frame_positions), dtype=torch.bool, device=camera_frame_positions.device)
        full_mask[depth_mask] = projection_mask
        
        return full_mask
