import torch
from typing import Any, Optional
from data.transforms.base_transform import BaseTransform
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.select import Select


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

    def __call__(
        self, pc: PointCloud, seed: Optional[Any] = None
    ) -> PointCloud:
        """
        Scale down point cloud and subsample points proportionally.

        Args:
            pc: Point cloud with required xyz field

        Returns:
            PointCloud with scaled and subsampled points

        Raises:
            ValueError: If the scale factor is too small, resulting in 0 points after scaling
        """
        assert isinstance(pc, PointCloud), f"{type(pc)=}"

        points = pc.xyz
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

        selected = Select(indices)(pc)
        selected.xyz = selected.xyz * self.scale_factor
        return selected

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale_factor={self.scale_factor})"
