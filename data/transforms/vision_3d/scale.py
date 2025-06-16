import torch
import numpy as np
from typing import Dict, Any, Optional
from data.transforms.base_transform import BaseTransform


class Scale(BaseTransform):
    """Scale transform that reduces point cloud size while maintaining density through proportional subsampling."""

    def __init__(self, scale_factor: float = 0.1):
        """
        Args:
            scale_factor: Factor to scale down the point cloud (e.g., 0.1 means reduce to 10% of original size)
        """
        super(Scale, self).__init__()
        assert isinstance(scale_factor, (int, float)), f"{type(scale_factor)=}"
        assert scale_factor > 0 and scale_factor < 1, f"{scale_factor=}"
        self.scale_factor = scale_factor

    def __call__(self, pc: Dict[str, Any], seed: Optional[Any] = None) -> Dict[str, Any]:
        """
        Scale down point cloud and subsample points proportionally.

        Args:
            pc: Dictionary containing point cloud data with 'pos' key

        Returns:
            Dictionary with scaled and subsampled point cloud

        Raises:
            ValueError: If the scale factor is too small, resulting in 0 points after scaling
        """
        assert isinstance(pc, dict), f"{type(pc)=}"
        assert 'pos' in pc, f"'pos' not found in {pc.keys()}"
        assert pc['pos'].shape[0] > 0, f"{pc['pos'].shape=}"
        assert all(isinstance(v, torch.Tensor) for v in pc.values()), \
            f"{[type(v) for v in pc.values()]=}"
        assert all(pc[k].shape[0] == pc['pos'].shape[0] for k in pc.keys() if k != 'pos'), \
            f"{pc['pos'].shape=}, {[pc[k].shape for k in pc.keys() if k != 'pos']=}"

        # Get points
        points = pc['pos']
        num_points = points.shape[0]
        device = points.device

        # Calculate number of points to keep based on scale factor
        # Since we're scaling in 3D, we need to reduce points by scale_factor^3
        # to maintain the same density
        target_points = int(num_points * (self.scale_factor ** 3))
        if target_points == 0:
            raise ValueError(f"Scale factor {self.scale_factor} is too small for point cloud with {num_points} points. "
                           f"Would result in 0 points after scaling.")

        # Randomly sample points
        generator = self._get_generator(g_type='torch', seed=seed)
        indices = torch.randperm(num_points, device=device, generator=generator)[:target_points]

        # Create new dictionary with scaled and subsampled values
        result = {}
        for key, value in pc.items():
            if key == 'pos':
                # For XYZ coordinates, both subsample and scale
                result[key] = value[indices] * self.scale_factor
            else:
                # For other features, only subsample
                result[key] = value[indices]

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale_factor={self.scale_factor})"
