from typing import Dict, Any, List, Union, Optional
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks import check_point_cloud
from utils.point_cloud_ops.select import Select


class Clamp(BaseTransform):

    def __init__(self, max_points: int) -> None:
        assert isinstance(max_points, int)
        assert max_points > 0, f"{max_points=}"
        self.max_points = max_points
        super(Clamp, self).__init__()

    def __call__(self, *args, seed: Optional[Any] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Apply clamp transform to one or more point clouds with consistent randomness.
        
        Args:
            *args: One or more point cloud dictionaries to clamp
            seed: Optional seed for reproducible results
            
        Returns:
            Single point cloud dict if one arg, list of point cloud dicts if multiple args
        """
        assert isinstance(args, tuple), f"{type(args)=}"
        assert len(args) > 0, f"{len(args)=}"
        
        # Validate all inputs are point clouds
        for i, pc in enumerate(args):
            assert isinstance(pc, dict), f"Argument {i} must be dict, got {type(pc)}"
            check_point_cloud(pc)
        
        # Check if all point clouds have the same number of points and device
        num_points_list = [pc['pos'].shape[0] for pc in args]
        devices = [pc['pos'].device for pc in args]
        
        # Ensure all point clouds have the same number of points for consistent clamping
        if len(set(num_points_list)) > 1:
            raise ValueError(f"All point clouds must have the same number of points for consistent clamping. Got: {num_points_list}")
        
        # Ensure all point clouds are on the same device
        if len(set(devices)) > 1:
            raise ValueError(f"All point clouds must be on the same device. Got: {devices}")
        
        num_points = num_points_list[0]
        device = devices[0]
        
        # If no clamping needed, return all point clouds unchanged
        if num_points <= self.max_points:
            return args[0] if len(args) == 1 else list(args)
        
        # Generate random indices once for consistent clamping across all point clouds
        generator = self._get_generator(g_type='torch', seed=seed)
        # Ensure generator is on the same device as point clouds
        if generator.device != device:
            # Create new generator on correct device with same seed
            generator = torch.Generator(device=device)
            if seed is None:
                import random
                seed = random.randint(0, 2**32 - 1)
            if not isinstance(seed, int):
                seed = hash(seed) % (2**32)
            generator.manual_seed(seed)
        
        indices = torch.randperm(num_points, generator=generator, device=device)[:self.max_points]
        
        # Apply the same indices to all point clouds
        result = [Select(indices)(pc) for pc in args]
        
        # Return single result if only one input, otherwise return list
        return result[0] if len(result) == 1 else result
