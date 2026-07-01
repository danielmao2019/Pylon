from typing import Dict, Any, Optional, Union
import torch
import numpy as np
from sklearn.neighbors import KDTree
from data.structures.three_d.point_cloud.point_cloud import PointCloud


class CylinderSampling:
    """Sample points within a cylinder.

    Args:
        radius: Radius of the cylinder.
        center: Center position of the cylinder base (3,).
        align_origin: Whether to align sampled points to the origin by
            subtracting the center coordinates.
        device: Optional device to place tensors on.

    Raises:
        ValueError: If radius is not positive.
    """

    def __init__(
        self,
        radius: float,
        center: Union[torch.Tensor, np.ndarray],
        align_origin: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        if radius <= 0:
            raise ValueError("Radius must be positive")

        self._radius = radius
        self._center = torch.as_tensor(center, device=device).view(1, -1)
        self._align_origin = align_origin

    def __call__(
        self, kdtree: KDTree, pc: PointCloud
    ) -> PointCloud:
        """Sample points within the cylinder.

        Args:
            kdtree: KDTree built from the points.
            pc: Point cloud containing points and attributes.
                May contain 'change_map' key for labels.

        Returns:
            PointCloud containing:
            - xyz: Sampled points (M, D)
            - point_idx: Indices of sampled points in original point cloud (M,)
            - change_map: Sampled labels if present in input (M,)

        """
        assert isinstance(pc, PointCloud), f"{type(pc)=}"

        # Add debug prints
        print(f"Cylinder sampling debug:")
        print(f"  Center: {self._center}")
        print(f"  Radius: {self._radius}")
        print(f"  Input points shape: {pc.xyz.shape}")
        print(
            f"  Input points bounds: min={pc.xyz.min(0)[0]}, max={pc.xyz.max(0)[0]}"
        )

        # Query points within radius - still need to convert to numpy for scikit-learn KDTree
        indices = torch.LongTensor(
            kdtree.query_radius(self._center.cpu().numpy(), r=self._radius)[0]
        )
        print(f"  Found {len(indices)} points within radius")

        if len(indices) == 0:
            print("  WARNING: No points found within cylinder radius!")
            print(f"  First few input points: {pc.xyz[:5]}")

        if self._center.device.type != 'cpu':
            indices = indices.to(self._center.device)

        # Sample position data
        pos = pc.xyz
        sampled_pos = pos[indices]
        if self._align_origin:
            sampled_pos = sampled_pos.clone()
            sampled_pos[:, : self._center.shape[1]] -= self._center

        result_fields: Dict[str, torch.Tensor] = {'point_idx': indices}

        if hasattr(pc, 'change_map'):
            result_fields['change_map'] = pc.change_map[indices]

        return PointCloud(xyz=sampled_pos, data=result_fields)

    def __repr__(self) -> str:
        return "{}(radius={}, center={}, align_origin={})".format(
            self.__class__.__name__,
            self._radius,
            self._center.tolist(),
            self._align_origin,
        )
