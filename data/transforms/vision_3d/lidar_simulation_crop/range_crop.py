from typing import Dict
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks.point_cloud import check_point_cloud


class RangeCrop(BaseTransform):
    """Range-based cropping for LiDAR sensor simulation.
    
    Filters points based on maximum sensor range, simulating the physical
    limitation that LiDAR sensors cannot detect objects beyond their range.
    """

    def __init__(self, max_range: float = 100.0):
        """Initialize range crop.
        
        Args:
            max_range: Maximum sensor range in meters (typical automotive: 100-200m)
        """
        assert isinstance(max_range, (int, float)), f"max_range must be numeric, got {type(max_range)}"
        assert max_range > 0, f"max_range must be positive, got {max_range}"
        
        self.max_range = float(max_range)

    def _call_single(self, pc: Dict[str, torch.Tensor], sensor_pos: torch.Tensor, 
                    *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Apply range cropping to point cloud.
        
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
        
        # Align sensor_pos to positions.device if needed
        if sensor_pos.device != positions.device:
            sensor_pos = sensor_pos.to(positions.device)
        
        # Calculate distances from sensor to all points
        distances = torch.norm(positions - sensor_pos.unsqueeze(0), dim=1)
        
        # Create range mask
        range_mask = distances <= self.max_range
        
        # Apply mask to all keys in point cloud
        cropped_pc = {}
        for key, tensor in pc.items():
            if key == 'pos':
                cropped_pc[key] = positions[range_mask]
            else:
                # Assume features have same first dimension as positions
                cropped_pc[key] = tensor[range_mask]
        
        return cropped_pc