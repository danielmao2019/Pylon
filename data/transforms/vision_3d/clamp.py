from typing import Any, List, Optional, Union
import torch
from data.transforms.base_transform import BaseTransform
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.select import Select


class Clamp(BaseTransform):

    def __init__(self, max_points: int) -> None:
        assert isinstance(max_points, int)
        assert max_points > 0, f"{max_points=}"
        self.max_points = max_points
        super(Clamp, self).__init__()

    def __call__(self, *args, seed: Optional[Any] = None) -> Union[PointCloud, List[PointCloud]]:
        """
        Apply clamp transform to one or more point clouds with consistent randomness.

        Args:
            *args: One or more point clouds to clamp
            seed: Optional seed for reproducible results

        Returns:
            Single PointCloud if one arg, list of PointClouds if multiple args
        """
        assert isinstance(args, tuple), f"{type(args)=}"
        assert len(args) > 0, f"{len(args)=}"

        point_clouds: List[PointCloud] = []
        for i, pc in enumerate(args):
            assert isinstance(pc, PointCloud), f"Argument {i} must be PointCloud, got {type(pc)}"
            point_clouds.append(pc)

        # Check if all point clouds have the same number of points and device
        num_points_list = [pc.xyz.shape[0] for pc in point_clouds]
        devices = [pc.xyz.device for pc in point_clouds]

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
            return point_clouds[0] if len(point_clouds) == 1 else list(point_clouds)

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
                from utils.determinism.hash_utils import convert_to_seed
                seed = convert_to_seed(seed)
            generator.manual_seed(seed)

        indices = torch.randperm(num_points, generator=generator, device=device)[:self.max_points]

        # Apply the same indices to all point clouds
        result = [Select(indices)(pc) for pc in point_clouds]

        # Return single result if only one input, otherwise return list
        return result[0] if len(result) == 1 else result
