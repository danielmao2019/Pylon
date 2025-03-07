from typing import Dict, Any, Optional
import torch
import numpy as np
from sklearn.neighbors import KDTree


def consecutive_cluster(src):
    """Convert a cluster index tensor to consecutive indices.
    
    Parameters
    ----------
    src : torch.Tensor
        Input cluster indices
        
    Returns
    -------
    inv : torch.Tensor
        Cluster indices converted to consecutive numbers
    perm : torch.Tensor
        Indices of unique elements in original ordering
    """
    unique, inv = torch.unique(src, sorted=True, return_inverse=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    return inv, perm


def grid_cluster(
    pos: torch.Tensor,
    size: torch.Tensor,
    start: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """A clustering algorithm, which overlays a regular grid of user-defined
    size over a point cloud and clusters all points within a voxel.

    Args:
        pos (Tensor): D-dimensional position of points
        size (Tensor): Size of a voxel in each dimension
        start (Tensor, optional): Start position of the grid (in each dimension)
        end (Tensor, optional): End position of the grid (in each dimension)

    Returns:
        Tensor: Cluster indices for each point
    """
    # If start/end not provided, compute them from point cloud bounds
    if start is None:
        start = pos.min(dim=0)[0]
    if end is None:
        end = pos.max(dim=0)[0]

    # Shift points to start at origin and get grid coordinates
    pos = pos - start
    
    # Get grid coordinates for each point using floor division
    grid_coords = torch.floor(pos / size).long()

    # Compute grid dimensions
    grid_size = torch.ceil((end - start) / size).long()
    
    # Normalize coordinates to be non-negative
    grid_coords = grid_coords - grid_coords.min(dim=0)[0]
    
    # Compute unique cell index using row-major ordering
    # This is more robust than Morton encoding for large coordinate ranges
    strides = torch.tensor([1, grid_size[0], grid_size[0] * grid_size[1]], 
                          device=pos.device, dtype=torch.long)
    cluster = (grid_coords * strides.view(1, -1)).sum(dim=1)
    
    return cluster


def group_data(data_dict: Dict[str, torch.Tensor], cluster=None, unique_pos_indices=None, mode="last"):
    """Group data based on indices in cluster.
    
    Parameters
    ----------
    data_dict : Dict[str, torch.Tensor]
        Dictionary containing tensors to be grouped
    cluster : torch.Tensor
        Cluster indices for each point
    unique_pos_indices : torch.Tensor
        Indices to select points for 'last' mode
    mode : str
        'mean' or 'last'
    """
    assert mode in ["mean", "last"]
    if mode == "mean" and cluster is None:
        raise ValueError("In mean mode the cluster argument needs to be specified")
    if mode == "last" and unique_pos_indices is None:
        raise ValueError("In last mode the unique_pos_indices argument needs to be specified")

    result_dict = {}
    
    # Handle positions (continuous data)
    pos = data_dict['pos']
    if mode == "last":
        result_dict['pos'] = pos[unique_pos_indices]
    else:  # mode == "mean"
        num_clusters = cluster.max().item() + 1
        summed = torch.zeros((num_clusters, pos.size(1)),
                          dtype=pos.dtype, device=pos.device)
        counts = torch.zeros(num_clusters, dtype=torch.float, device=pos.device)
        for i in range(pos.size(0)):
            summed[cluster[i]] += pos[i]
            counts[cluster[i]] += 1
        result_dict['pos'] = summed / counts.unsqueeze(1)
    
    # Handle change map if present (categorical data)
    if 'change_map' in data_dict:
        change_map = data_dict['change_map']
        if mode == "last":
            result_dict['change_map'] = change_map[unique_pos_indices]
        else:  # mode == "mean"
            num_clusters = cluster.max().item() + 1
            change_min = change_map.min()
            one_hot = torch.zeros((change_map.size(0), change_map.max() - change_min + 1),
                               device=change_map.device)
            one_hot.scatter_(1, (change_map - change_min).unsqueeze(1), 1)
            summed = torch.zeros((num_clusters, one_hot.size(1)),
                              device=change_map.device)
            for i in range(change_map.size(0)):
                summed[cluster[i]] += one_hot[i]
            result_dict['change_map'] = summed.argmax(dim=1) + change_min

    return result_dict


class GridSampling3D:
    """Clusters points into voxels with specified size."""

    def __init__(self, size, mode="mean"):
        self._grid_size = size
        self._mode = mode
        if mode not in ["mean", "last"]:
            raise ValueError(f"Mode {mode} not supported. Use 'mean' or 'last'")

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Sample points and associated data by grouping them into voxels.

        Parameters
        ----------
        data_dict : Dict[str, torch.Tensor]
            Dictionary containing points and attributes

        Returns
        -------
        Dict[str, torch.Tensor]
            Sampled data dictionary
        """
        if 'pos' not in data_dict:
            raise ValueError("Data dictionary must have 'pos' key")
        
        points = data_dict['pos']
        if points.shape[1] < 3:
            raise ValueError("Points must have at least 3 dimensions (x, y, z)")

        # Get cluster indices using grid_cluster
        cluster = grid_cluster(
            points[:, :3],
            torch.tensor([self._grid_size, self._grid_size, self._grid_size], 
                        dtype=points.dtype, 
                        device=points.device)
        )
        
        # Get consecutive cluster indices and permutation
        cluster, unique_pos_indices = consecutive_cluster(cluster)
        
        # Group all data attributes
        return group_data(data_dict, cluster, unique_pos_indices, mode=self._mode)

    def __repr__(self):
        return "{}(grid_size={}, mode={})".format(
            self.__class__.__name__, self._grid_size, self._mode
        )
