from typing import Dict, Union
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
    """

    def __init__(self, radius: float, center: Union[torch.Tensor, np.ndarray], align_origin: bool = True) -> None:
        self._radius = radius
        self._center = torch.as_tensor(center).view(1, -1)
        self._align_origin = align_origin

    def __call__(self, kdtree: KDTree, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

        # Query points within radius - still need to convert to numpy for scikit-learn KDTree
        indices = torch.LongTensor(kdtree.query_radius(self._center.numpy(), r=self._radius)[0])
        
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
            self.__class__.__name__, self._radius, self._center, self._align_origin
        )
