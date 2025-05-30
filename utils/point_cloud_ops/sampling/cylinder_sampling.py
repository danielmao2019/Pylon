from typing import Dict, Union, Any, Optional
import torch
import numpy as np
from sklearn.neighbors import KDTree


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
        device: Optional[torch.device] = None
    ) -> None:
        if radius <= 0:
            raise ValueError("Radius must be positive")

        self._radius = radius
        self._center = torch.as_tensor(center, device=device).view(1, -1)
        self._align_origin = align_origin

    def __call__(self, kdtree: KDTree, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Sample points within the cylinder.

        Args:
            kdtree: KDTree built from the points.
            data_dict: Dictionary containing points and attributes.
                Must contain 'pos' key with points of shape (N, D).
                May contain 'change_map' key for labels.

        Returns:
            Dictionary containing:
            - pos: Sampled points (M, D)
            - point_idx: Indices of sampled points in original point cloud (M,)
            - change_map: Sampled labels if present in input (M,)

        Raises:
            ValueError: If 'pos' key is missing from data_dict.
        """
        if 'pos' not in data_dict:
            raise ValueError("Data dictionary must have 'pos' key")

        # Add debug prints
        print(f"Cylinder sampling debug:")
        print(f"  Center: {self._center}")
        print(f"  Radius: {self._radius}")
        print(f"  Input points shape: {data_dict['pos'].shape}")
        print(f"  Input points bounds: min={data_dict['pos'].min(0)[0]}, max={data_dict['pos'].max(0)[0]}")

        # Query points within radius - still need to convert to numpy for scikit-learn KDTree
        indices = torch.LongTensor(kdtree.query_radius(self._center.cpu().numpy(), r=self._radius)[0])
        print(f"  Found {len(indices)} points within radius")

        if len(indices) == 0:
            print("  WARNING: No points found within cylinder radius!")
            print(f"  First few input points: {data_dict['pos'][:5]}")

        if self._center.device.type != 'cpu':
            indices = indices.to(self._center.device)

        # Sample position data
        pos = data_dict['pos']
        sampled_pos = pos[indices]
        if self._align_origin:
            sampled_pos = sampled_pos.clone()
            sampled_pos[:, :self._center.shape[1]] -= self._center

        result_dict = {
            'pos': sampled_pos,
            'point_idx': indices
        }

        # Sample change map if present
        if 'change_map' in data_dict:
            result_dict['change_map'] = data_dict['change_map'][indices]

        return result_dict

    def __repr__(self) -> str:
        return "{}(radius={}, center={}, align_origin={})".format(
            self.__class__.__name__, self._radius, self._center.tolist(), self._align_origin
        )
