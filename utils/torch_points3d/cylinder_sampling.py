from typing import Dict, Union
import torch
import numpy as np
from sklearn.neighbors import KDTree


class CylinderSampling:
    """Sample points within a cylinder."""

    def __init__(self, radius: float, center: Union[torch.Tensor, np.ndarray], align_origin: bool = True) -> None:
        self._radius = radius
        self._center = np.asarray(center).reshape(1, -1)
        self._align_origin = align_origin

    def __call__(self, kdtree: KDTree, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Sample points within the cylinder.

        Parameters
        ----------
        kdtree : KDTree
            KDTree built from the points
        data_dict : Dict[str, torch.Tensor]
            Dictionary containing points and attributes

        Returns
        -------
        Dict[str, torch.Tensor]
            Sampled data dictionary
        """
        if 'pos' not in data_dict:
            raise ValueError("Data dictionary must have 'pos' key")

        # Query points within radius
        indices = torch.LongTensor(kdtree.query_radius(self._center, r=self._radius)[0])
        
        # Sample position data
        pos = data_dict['pos']
        sampled_pos = pos[indices]
        if self._align_origin:
            t_center = torch.FloatTensor(self._center)
            sampled_pos = sampled_pos.clone()
            sampled_pos[:, :self._center.shape[1]] -= t_center
        
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
