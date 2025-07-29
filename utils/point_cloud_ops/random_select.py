from typing import Any, Dict, Optional
import torch
from utils.input_checks import check_point_cloud
from utils.point_cloud_ops.select import Select


class RandomSelect:
    def __init__(self, percentage: Optional[float] = None, count: Optional[int] = None) -> None:
        """Randomly select a subset of points from a point cloud.
        
        Args:
            percentage: Fraction of points to select (0 < percentage <= 1). Mutually exclusive with count.
            count: Exact number of points to select (count > 0). Mutually exclusive with percentage.
        """
        # XOR logic: exactly one of percentage or count must be provided
        assert (percentage is not None) ^ (count is not None), \
            f"Exactly one of percentage or count must be provided, got percentage={percentage}, count={count}"
        
        if percentage is not None:
            assert isinstance(percentage, (int, float)), f"{type(percentage)=}"
            assert 0 < percentage <= 1, f"{percentage=}"
            self.percentage = percentage
            self.count = None
        else:
            assert isinstance(count, int), f"{type(count)=}"
            assert count > 0, f"{count=}"
            self.percentage = None
            self.count = count

    def __call__(self, pc: Dict[str, Any], seed: Optional[Any] = None, generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
        check_point_cloud(pc)
        
        # XOR logic: exactly one of seed or generator must be provided
        assert (seed is not None) ^ (generator is not None), \
            f"Exactly one of seed or generator must be provided, got seed={seed}, generator={generator}"
        
        # Create or use generator
        if generator is not None:
            # Validate generator device type matches point cloud device type
            assert generator.device.type == pc['pos'].device.type, (
                f"Generator device type '{generator.device.type}' must match point cloud device type '{pc['pos'].device.type}'"
            )
            gen = generator
        else:
            # Create generator on the same device as point cloud
            gen = torch.Generator(device=pc['pos'].device)
            # Handle tuple seeds like BaseTransform does
            if not isinstance(seed, int):
                seed = hash(seed) % (2**32)  # Ensure it's a 32-bit integer
            gen.manual_seed(seed)
        
        num_points = pc['pos'].shape[0]
        
        if self.percentage is not None:
            num_points_to_select = int(num_points * self.percentage)
        else:
            num_points_to_select = min(self.count, num_points)  # Don't exceed available points
        
        # Use generator for deterministic random selection
        indices = torch.randperm(num_points, generator=gen, device=pc['pos'].device)[:num_points_to_select]
        return Select(indices)(pc)

    def __str__(self) -> str:
        """String representation of the RandomSelect transform."""
        if self.percentage is not None:
            return f"RandomSelect(percentage={self.percentage})"
        else:
            return f"RandomSelect(count={self.count})"
