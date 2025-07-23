from typing import Dict, Union, Tuple
import torch
from abc import ABC, abstractmethod
from data.transforms.base_transform import BaseTransform
from utils.input_checks.point_cloud import check_point_cloud


class BaseFOVCrop(BaseTransform, ABC):
    """Base class for field-of-view cropping transforms.
    
    This abstract base class defines the common interface and shared functionality
    for different FOV cropping modes:
    - SphericalFOVCrop: LiDAR-style cone-shaped cropping using spherical coordinates
    - PerspectiveFOVCrop: Camera-style rectangular frustum cropping using perspective projection
    
    Both modes support the same FOV parameter format but apply different geometric constraints.
    """

    def __init__(
        self,
        fov: Tuple[Union[int, float], Union[int, float]]
    ):
        """Initialize base FOV crop.
        
        Args:
            fov: Tuple of (horizontal_fov, vertical_fov) in degrees
                - horizontal_fov: Horizontal field of view total angle 
                - vertical_fov: Vertical field of view total angle
        """
        # Validate FOV parameter
        assert isinstance(fov, tuple), f"fov must be tuple, got {type(fov)}"
        assert len(fov) == 2, f"fov must be tuple of length 2, got length {len(fov)}"
        horizontal_fov, vertical_fov = fov
        
        assert isinstance(horizontal_fov, (int, float)), f"horizontal_fov must be numeric, got {type(horizontal_fov)}"
        assert isinstance(vertical_fov, (int, float)), f"vertical_fov must be numeric, got {type(vertical_fov)}"
        
        # Validate FOV ranges - subclasses can override these limits
        self._validate_fov_ranges(horizontal_fov, vertical_fov)
        
        # Store FOV as tuple (consistent storage format)
        self.fov = (float(horizontal_fov), float(vertical_fov))
        
        # Also provide individual access for backward compatibility and convenience
        self.horizontal_fov = float(horizontal_fov)
        self.vertical_fov = float(vertical_fov)

    def _validate_fov_ranges(self, horizontal_fov: float, vertical_fov: float) -> None:
        """Validate FOV ranges - can be overridden by subclasses.
        
        Args:
            horizontal_fov: Horizontal FOV in degrees
            vertical_fov: Vertical FOV in degrees
        """
        # Default validation - subclasses can override for specific constraints
        assert 0 < horizontal_fov <= 360, f"horizontal_fov must be in (0, 360], got {horizontal_fov}"
        assert 0 < vertical_fov <= 180, f"vertical_fov must be in (0, 180], got {vertical_fov}"

    def _call_single(self, pc: Dict[str, torch.Tensor], sensor_extrinsics: torch.Tensor,
                    *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Apply FOV cropping to point cloud.
        
        Args:
            pc: Point cloud dictionary with 'pos' key and optional feature keys
            sensor_extrinsics: 4x4 sensor pose matrix (sensor-to-world transform)
            
        Returns:
            Cropped point cloud dictionary
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
        
        # Apply FOV constraints (implemented by subclasses)
        fov_mask = self._apply_fov_constraints(sensor_frame_positions)
        
        # Apply mask to all keys in point cloud
        cropped_pc = {}
        for key, tensor in pc.items():
            if key == 'pos':
                cropped_pc[key] = positions[fov_mask]
            else:
                # Assume features have same first dimension as positions
                cropped_pc[key] = tensor[fov_mask]
        
        return cropped_pc

    @abstractmethod
    def _apply_fov_constraints(self, sensor_frame_positions: torch.Tensor) -> torch.Tensor:
        """Apply field-of-view constraints to points in sensor coordinate frame.
        
        This method must be implemented by subclasses to define the specific
        geometric constraints for their FOV cropping mode.
        
        Args:
            sensor_frame_positions: Point positions in sensor coordinate frame [N, 3]
            
        Returns:
            Boolean mask [N] indicating points within FOV
        """
        pass
