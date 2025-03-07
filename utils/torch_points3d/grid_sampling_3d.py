from typing import Optional
import torch


def grid_cluster(
    pos: torch.Tensor,
    size: torch.Tensor,
    start: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """A clustering algorithm, which overlays a regular grid of user-defined
    size over a point cloud and clusters all points within a voxel.

    Args:
        pos (Tensor): D-dimensional position of points.
        size (Tensor): Size of a voxel in each dimension.
        start (Tensor, optional): Start position of the grid (in each
            dimension). (default: :obj:`None`)
        end (Tensor, optional): End position of the grid (in each
            dimension). (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import grid_cluster

        pos = torch.Tensor([[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]])
        size = torch.Tensor([5, 5])
        cluster = grid_cluster(pos, size)
    """
    return torch.ops.torch_cluster.grid(pos, size, start, end)


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


def group_data(points, cluster, unique_pos_indices, mode="mean"):
    """Group points based on cluster indices.
    
    Parameters
    ----------
    points : torch.Tensor
        Input points
    cluster : torch.Tensor
        Cluster indices for each point
    unique_pos_indices : torch.Tensor
        Indices to select points for 'last' mode
    mode : str
        'mean' or 'last'
        
    Returns
    -------
    torch.Tensor
        Grouped points
    """
    if mode == "last":
        return points[unique_pos_indices]
    elif mode == "mean":
        # Manual mean computation using scatter_add
        num_clusters = cluster.max().item() + 1
        sum_points = torch.zeros((num_clusters, points.shape[1]), dtype=points.dtype, device=points.device)
        counts = torch.zeros(num_clusters, dtype=points.dtype, device=points.device)
        
        # Sum points in each cluster
        for i in range(points.shape[0]):
            c = cluster[i]
            sum_points[c] += points[i]
            counts[c] += 1
            
        # Compute means
        means = sum_points / counts.unsqueeze(1)
        return means
    else:
        raise ValueError(f"Mode {mode} not supported")


class GridSampling3D:
    """Clusters points into voxels with size :attr:`size`.
    
    Parameters
    ----------
    size: float
        Size of a voxel (in each dimension).
    mode: string
        The mode can be either `last` or `mean`.
        If mode is `mean`, all the points within a cell will be averaged
        If mode is `last`, one point per cell will be selected
    """

    def __init__(self, size, mode="mean"):
        self._grid_size = size
        self._mode = mode
        if mode not in ["mean", "last"]:
            raise ValueError(f"Mode {mode} not supported. Use 'mean' or 'last'")

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """Sample points by grouping them into voxels.

        Parameters
        ----------
        points : torch.Tensor
            Input points of shape (N, 3) or (N, D) where D >= 3

        Returns
        -------
        torch.Tensor
            Sampled points
        """
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points)
        
        if points.shape[1] < 3:
            raise ValueError("Points must have at least 3 dimensions (x, y, z)")

        # Round coordinates to grid
        coords = points[:, :3] / self._grid_size
        
        # Get cluster indices using grid_cluster
        cluster = grid_cluster(coords, torch.tensor([1., 1., 1.]))
        
        # Get consecutive cluster indices and permutation
        cluster, unique_pos_indices = consecutive_cluster(cluster)
        
        # Group points using the provided logic
        sampled_points = group_data(points, cluster, unique_pos_indices, mode=self._mode)
        
        return sampled_points

    def __repr__(self):
        return "{}(grid_size={}, mode={})".format(
            self.__class__.__name__, self._grid_size, self._mode
        )
