from typing import Dict, Any, Optional, Tuple, Union, List
import torch


def consecutive_cluster(src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a cluster index tensor to consecutive indices.

    Args:
        src: Input cluster indices tensor.

    Returns:
        A tuple containing:
        - inv: Cluster indices converted to consecutive numbers
        - perm: Indices of unique elements in original ordering
    """
    # Match original implementation by using sorted=True
    unique, inv = torch.unique(src, sorted=True, return_inverse=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    return inv, perm


def grid_cluster(
    pos: torch.Tensor,
    size: Union[float, List[float], torch.Tensor],
    start: Optional[Union[List[float], torch.Tensor]] = None,
    end: Optional[Union[List[float], torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute cluster indices for grid sampling.

    Args:
        pos: Point positions tensor of shape (N, 3)
        size: Voxel size(s) for grid sampling
        start: Optional start position for grid
        end: Optional end position for grid

    Returns:
        Cluster indices tensor of shape (N,)
    """
    # Convert size to tensor if needed
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=pos.device)
    if size.dim() == 0:
        size = size.repeat(3)

    # Compute grid coordinates
    if start is not None:
        if not isinstance(start, torch.Tensor):
            start = torch.tensor(start, device=pos.device)
        # Match original implementation by modifying pos directly
        pos = pos - start

    # Compute grid indices
    grid_coords = torch.floor(pos / size).long()

    # Normalize coordinates to start from 0
    grid_coords = grid_coords - grid_coords.min(dim=0)[0]

    # Compute unique grid indices
    grid_indices = grid_coords[:, 0] + grid_coords[:, 1] * grid_coords[:, 2].max() + grid_coords[:, 2] * grid_coords[:, 1].max() * grid_coords[:, 0].max()

    return grid_indices


def compute_grid_indices(pos: torch.Tensor, size: float) -> torch.Tensor:
    """Compute grid indices for each point.

    Args:
        pos: Point positions (N, 3).
        size: Size of voxel cells.

    Returns:
        Grid indices for each point (N, 3).
    """
    # Compute grid indices using floor division
    grid_indices = torch.floor(pos / size).long()
    return grid_indices


class GridSampling3Dv2:
    """Clusters points into voxels with specified size.

    Args:
        size: Size of voxels in each dimension.
        mode: Grouping mode, either 'mean' or 'last'.
            - 'mean': Average points within each voxel
            - 'last': Select last point from each voxel
        device: Optional device to place tensors on.

    Raises:
        ValueError: If size is not positive or mode is invalid.
    """

    def __init__(
        self,
        size: float,
        mode: str = "mean",
        device: Optional[torch.device] = None
    ) -> None:
        if size <= 0:
            raise ValueError("Size must be positive")
        if mode not in ["mean", "last"]:
            raise ValueError(f"Mode {mode} not supported. Use 'mean' or 'last'")

        self._grid_size = size
        self._mode = mode
        self._device = device
        
        # Cache for frequently used tensors
        self._size_tensor = None
        self._strides = None

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Sample points and associated data by grouping them into voxels.

        Args:
            data_dict: Dictionary containing points and attributes.
                Must contain 'pos' key with points of shape (N, D), D >= 3.

        Returns:
            Dictionary containing sampled data with same keys as input.
            Points are grouped into voxels according to the specified mode.

        Raises:
            ValueError: If 'pos' key is missing or points have less than 3 dimensions.
        """
        if 'pos' not in data_dict:
            raise ValueError("Data dictionary must have 'pos' key")

        points = data_dict['pos']
        if points.shape[1] < 3:
            raise ValueError("Points must have at least 3 dimensions (x, y, z)")

        # Get cluster indices using grid_cluster
        # Optimized: Cache size_tensor to avoid recreating it
        if self._size_tensor is None or self._size_tensor.device != points.device:
            self._size_tensor = torch.tensor(
                [self._grid_size, self._grid_size, self._grid_size],
                dtype=points.dtype,
                device=points.device if self._device is None else self._device
            )
        
        cluster = grid_cluster(points[:, :3], self._size_tensor)

        # Get consecutive cluster indices and permutation
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        # Group all data attributes based on the mode
        if self._mode == "mean":
            return self.group_data_mean(data_dict, cluster)
        else:  # mode == "last"
            return self.group_data_last(data_dict, unique_pos_indices)

    def group_data_mean(
        self,
        data_dict: Dict[str, torch.Tensor],
        cluster: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Group data by averaging points within each cluster.

        Args:
            data_dict: Dictionary containing tensors to be grouped.
                Must contain 'pos' and optionally 'change_map'.
            cluster: Cluster indices for each point (N,).

        Returns:
            Dictionary containing grouped tensors with same keys as input.
        """
        result_dict = {}

        # Handle positions (continuous data)
        pos = data_dict['pos']
        num_clusters = cluster.max().item() + 1
        
        # Initialize tensors for accumulation
        summed = torch.zeros((num_clusters, pos.size(1)), dtype=pos.dtype, device=pos.device)
        counts = torch.zeros(num_clusters, dtype=torch.float, device=pos.device)
        
        # Accumulate positions for each cluster
        for i in range(pos.size(1)):
            summed[:, i].scatter_add_(0, cluster, pos[:, i])
        
        # Count points in each cluster
        counts.scatter_add_(0, cluster, torch.ones_like(cluster, dtype=torch.float))
        
        # Compute mean positions
        result_dict['pos'] = summed / counts.unsqueeze(1)

        # Handle change map (categorical data)
        if 'change_map' in data_dict:
            change_map = data_dict['change_map']
            num_clusters = cluster.max().item() + 1
            change_min = change_map.min()
            
            # Create one-hot encoding
            one_hot = torch.zeros((change_map.size(0), change_map.max() - change_min + 1),
                               device=change_map.device)
            one_hot.scatter_(1, (change_map - change_min).unsqueeze(1), 1)
            
            # Accumulate one-hot encodings for each cluster
            summed = torch.zeros((num_clusters, one_hot.size(1)), device=change_map.device)
            for i in range(one_hot.size(1)):
                summed[:, i].scatter_add_(0, cluster, one_hot[:, i])
            
            # Get the most common category in each cluster
            result_dict['change_map'] = summed.argmax(dim=1) + change_min
        else:
            result_dict['change_map'] = None

        # Handle point indices
        result_dict['point_indices'] = cluster

        return result_dict

    def group_data_last(
        self,
        data_dict: Dict[str, torch.Tensor],
        unique_pos_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Group data by selecting the last point from each cluster.

        Args:
            data_dict: Dictionary containing tensors to be grouped.
                Must contain 'pos' and optionally 'change_map'.
            unique_pos_indices: Indices to select points (M,).

        Returns:
            Dictionary containing grouped tensors with same keys as input.
        """
        result_dict = {}

        # Handle positions (continuous data)
        pos = data_dict['pos']
        result_dict['pos'] = pos[unique_pos_indices]

        # Handle change map (categorical data)
        if 'change_map' in data_dict:
            change_map = data_dict['change_map']
            result_dict['change_map'] = change_map[unique_pos_indices]
        else:
            result_dict['change_map'] = None

        # Handle point indices
        result_dict['point_indices'] = unique_pos_indices

        return result_dict

    def __repr__(self) -> str:
        return "{}(grid_size={}, mode={})".format(
            self.__class__.__name__, self._grid_size, self._mode
        )


def group_data(
    cluster: torch.Tensor,
    data: Dict[str, torch.Tensor],
    mode: str = "mean",
    unique_indices: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Group data by cluster indices.

    Args:
        cluster: Cluster indices tensor of shape (N,)
        data: Dictionary of tensors to group
        mode: Aggregation mode ("mean" or "last")
        unique_indices: Optional pre-computed unique cluster indices

    Returns:
        Dictionary of grouped tensors
    """
    # Get unique cluster indices if not provided
    if unique_indices is None:
        unique_indices = torch.unique(cluster, sorted=True)

    # Initialize output dictionary
    out = {}

    # Process each tensor in the data dictionary
    for key, src in data.items():
        # Handle categorical data (last dimension is number of categories)
        if key == "cat" and src.dim() > 1:
            # For categorical data, use mode aggregation
            out[key] = scatter(src, cluster, dim=0, reduce="max")
        else:
            # For other data, use specified mode
            out[key] = scatter(src, cluster, dim=0, reduce=mode)

    return out


def grid_sampling_3d(
    pos: torch.Tensor,
    data: Dict[str, torch.Tensor],
    size: Union[float, List[float], torch.Tensor],
    start: Optional[Union[List[float], torch.Tensor]] = None,
    end: Optional[Union[List[float], torch.Tensor]] = None,
    mode: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Grid sampling in 3D space.

    Args:
        pos: Point positions tensor of shape (N, 3)
        data: Dictionary of tensors to sample
        size: Voxel size (float, list of 3 floats, or tensor of shape (3,))
        start: Optional start position of the grid
        end: Optional end position of the grid
        mode: Aggregation mode ("mean" or "last")

    Returns:
        Dictionary of sampled tensors
    """
    # Compute cluster indices
    cluster = grid_cluster(pos, size, start, end)

    # Group data by cluster
    out = group_data(cluster, data, mode)

    return out
