from typing import Dict, Tuple
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks.point_cloud import check_point_cloud
from .range_crop import RangeCrop
from .fov_crop import FOVCrop
from .occlusion_crop import OcclusionCrop


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
        horizontal_fov: float = 360.0,
        vertical_fov: Tuple[float, float] = (-30.0, 10.0),
        apply_range_filter: bool = True,
        apply_fov_filter: bool = True, 
        apply_occlusion_filter: bool = False,
        ray_density_factor: float = 0.8
    ):
        """Initialize LiDAR simulation crop transform.
        
        Args:
            max_range: Maximum sensor range in meters (typical automotive: 100-200m)
            horizontal_fov: Horizontal field of view in degrees (360° for spinning, ~120° for solid-state)
            vertical_fov: Vertical FOV as (min_elevation, max_elevation) in degrees
            apply_range_filter: Whether to apply range-based filtering
            apply_fov_filter: Whether to apply field-of-view filtering
            apply_occlusion_filter: Whether to apply occlusion simulation (ray-casting)
            ray_density_factor: Fraction of ray length to check for occlusion (0.8 = check 80%)
        """
        # Validate inputs
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
        
        assert isinstance(ray_density_factor, (int, float)), f"ray_density_factor must be numeric, got {type(ray_density_factor)}"
        assert 0.1 <= ray_density_factor <= 1.0, f"ray_density_factor must be in [0.1, 1.0], got {ray_density_factor}"
        
        # Store parameters
        self.max_range = float(max_range)
        self.horizontal_fov = float(horizontal_fov)
        self.vertical_fov = tuple(vertical_fov)
        self.apply_range_filter = apply_range_filter
        self.apply_fov_filter = apply_fov_filter
        self.apply_occlusion_filter = apply_occlusion_filter
        self.ray_density_factor = float(ray_density_factor)
        
        # Initialize crop components
        self.range_crop = RangeCrop(max_range=max_range) if apply_range_filter else None
        self.fov_crop = FOVCrop(horizontal_fov=horizontal_fov, vertical_fov=vertical_fov) if apply_fov_filter else None
        self.occlusion_crop = OcclusionCrop(ray_density_factor=ray_density_factor) if apply_occlusion_filter else None

    def _call_single(self, pc: Dict[str, torch.Tensor], sensor_extrinsics: torch.Tensor, 
                    *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Apply LiDAR simulation crop to point cloud.
        
        Args:
            pc: Point cloud dictionary with 'pos' key and optional feature keys
            sensor_extrinsics: 4x4 sensor pose matrix (world-to-sensor transform)
            
        Returns:
            Cropped point cloud dictionary
        """
        check_point_cloud(pc)
        
        # Validate sensor extrinsics matrix
        assert isinstance(sensor_extrinsics, torch.Tensor), f"sensor_extrinsics must be torch.Tensor, got {type(sensor_extrinsics)}"
        assert sensor_extrinsics.shape == (4, 4), f"sensor_extrinsics must be 4x4, got {sensor_extrinsics.shape}"
        
        # Extract sensor position from extrinsics matrix
        sensor_pos = sensor_extrinsics[:3, 3]  # Translation component
        
        # Start with the original point cloud
        current_pc = pc
        
        # Apply crops in order: range -> FOV -> occlusion
        # This order is optimized for performance (cheapest crops first)
        
        if self.apply_range_filter:
            current_pc = self.range_crop._call_single(current_pc, sensor_pos)
        
        if self.apply_fov_filter:
            current_pc = self.fov_crop._call_single(current_pc, sensor_extrinsics)
        
        # Apply occlusion crop last (most expensive computation)
        if self.apply_occlusion_filter:
            current_pc = self.occlusion_crop._call_single(current_pc, sensor_pos)
        
        return current_pc
