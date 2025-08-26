from typing import Tuple, Optional
import torch


def _knn_torch(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: Optional[int] = None,
    radius: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch backend - memory intensive but works everywhere."""
    assert isinstance(query_points, torch.Tensor), "query_points must be torch.Tensor"
    assert isinstance(reference_points, torch.Tensor), "reference_points must be torch.Tensor"
    assert (k is None) != (radius is None), "Exactly one of k or radius must be specified"
    assert k is None or k > 0, f"k must be positive if provided, got {k}"
    assert query_points.shape[1] == 3, f"query_points must have 3 coordinates"
    assert reference_points.shape[1] == 3, f"reference_points must have 3 coordinates"
    
    if radius is not None:
        return _knn_torch_with_r(
            query_points=query_points,
            reference_points=reference_points,
            radius=radius
        )
    else:
        assert k is not None  # This assertion ensures k is not None for type checking
        return _knn_torch_with_k(
            query_points=query_points,
            reference_points=reference_points,
            k=k
        )


def _knn_torch_with_r(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    radius: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch backend for radius search - memory intensive but works everywhere."""
    assert isinstance(query_points, torch.Tensor), "query_points must be torch.Tensor"
    assert isinstance(reference_points, torch.Tensor), "reference_points must be torch.Tensor"
    assert isinstance(radius, (int, float)), f"radius must be numeric, got {type(radius)}"
    assert radius > 0, f"radius must be positive, got {radius}"
    assert query_points.shape[1] == 3, f"query_points must have 3 coordinates"
    assert reference_points.shape[1] == 3, f"reference_points must have 3 coordinates"
    
    distances_squared = _compute_distances_squared(
        query_points=query_points,
        reference_points=reference_points
    )
    
    # Mask out points beyond radius (set to inf)
    radius_mask = distances_squared > (radius * radius)
    distances_squared = distances_squared.masked_fill(radius_mask, float('inf'))
    
    # For radius search, preserve original order
    distances = distances_squared.sqrt()
    
    # Find valid neighbors and pack efficiently
    valid_mask = distances != float('inf')
    valid_per_query = valid_mask.sum(dim=1)
    max_neighbors = valid_per_query.max().item()
    if max_neighbors == 0:
        max_neighbors = 1
    
    # Create output tensors
    n_queries = query_points.shape[0]
    dist_out = torch.full(
        (n_queries, max_neighbors),
        float('inf'),
        dtype=query_points.dtype,
        device=query_points.device
    )
    idx_out = torch.full(
        (n_queries, max_neighbors),
        -1,
        dtype=torch.long,
        device=query_points.device
    )
    
    # Get valid positions and pack directly
    query_idx, ref_idx = torch.where(valid_mask)
    output_cols = (valid_mask.cumsum(dim=1) - 1)[valid_mask]
    
    # Fill output arrays
    dist_out[query_idx, output_cols] = distances[query_idx, ref_idx]
    idx_out[query_idx, output_cols] = ref_idx
    
    return dist_out, idx_out


def _knn_torch_with_k(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch backend for k-NN search - memory intensive but works everywhere."""
    assert isinstance(query_points, torch.Tensor), "query_points must be torch.Tensor"
    assert isinstance(reference_points, torch.Tensor), "reference_points must be torch.Tensor"
    assert k > 0, f"k must be positive, got {k}"
    assert query_points.shape[1] == 3, f"query_points must have 3 coordinates"
    assert reference_points.shape[1] == 3, f"reference_points must have 3 coordinates"
    
    distances_squared = _compute_distances_squared(
        query_points=query_points,
        reference_points=reference_points
    )
    
    k_actual = min(k, reference_points.shape[0])
    
    # Find k nearest neighbors
    if k_actual < reference_points.shape[0] // 10:
        # Use topk for small k (more efficient)
        dist_k, idx_k = distances_squared.topk(
            k_actual,
            dim=1,
            largest=False,
            sorted=True
        )
    else:
        # Use sort for large k
        sorted_dists, sorted_indices = distances_squared.sort(dim=1)
        dist_k = sorted_dists[:, :k_actual]
        idx_k = sorted_indices[:, :k_actual]
    
    # If k > actual results, pad with inf/-1
    if k > k_actual:
        n_queries = query_points.shape[0]
        dist_out = torch.full(
            (n_queries, k),
            float('inf'),
            dtype=query_points.dtype,
            device=query_points.device
        )
        idx_out = torch.full(
            (n_queries, k),
            -1,
            dtype=torch.long,
            device=query_points.device
        )
        
        distances = dist_k.sqrt()
        dist_out[:, :k_actual] = distances
        # Mask out indices where distance is inf (beyond radius)
        inf_mask = distances == float('inf')
        idx_k_masked = idx_k.masked_fill(inf_mask, -1)
        idx_out[:, :k_actual] = idx_k_masked
        
        return dist_out, idx_out
    else:
        # Convert to actual distances
        distances = dist_k.sqrt()
        # Mask out indices where distance is inf (beyond radius)
        inf_mask = distances == float('inf')
        idx_k_masked = idx_k.masked_fill(inf_mask, -1)
        return distances, idx_k_masked


def _compute_distances_squared(
    query_points: torch.Tensor,
    reference_points: torch.Tensor
) -> torch.Tensor:
    """Compute squared L2 distances between query and reference points."""
    # Efficient L2 distance computation using the identity:
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    
    # Compute squared norms
    query_norm = (query_points ** 2).sum(dim=1, keepdim=True)  # [N, 1]
    reference_norm = (reference_points ** 2).sum(dim=1, keepdim=True)  # [M, 1]
    
    assert query_norm.shape == (query_points.shape[0], 1), f"Wrong query_norm shape: {query_norm.shape}"
    assert reference_norm.shape == (reference_points.shape[0], 1), f"Wrong reference_norm shape: {reference_norm.shape}"
    
    # Compute dot products
    dot_product = torch.mm(query_points, reference_points.t())  # [N, M]
    assert dot_product.shape == (query_points.shape[0], reference_points.shape[0]), f"Wrong dot_product shape: {dot_product.shape}"
    
    # Compute squared distances
    distances_squared = query_norm + reference_norm.t() - 2 * dot_product  # [N, M]
    distances_squared = distances_squared.clamp_min(0)  # Numerical stability
    
    assert distances_squared.shape == (query_points.shape[0], reference_points.shape[0]), f"Wrong distances_squared shape: {distances_squared.shape}"
    assert (distances_squared >= 0).all(), "Distances squared must be non-negative"
    
    return distances_squared
