from typing import Tuple, Optional, Union
import torch
import numpy as np
from utils.input_checks.point_cloud import check_pc_xyz


def knn(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: Optional[int] = None,
    method: str = "faiss",
    return_distances: bool = True,
    radius: Optional[float] = None
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Compute k-nearest neighbors using various backends.
    
    Args:
        query_points: Query point cloud [N, D] where D is dimension (usually 3)
        reference_points: Reference point cloud [M, D]
        k: Number of nearest neighbors to find. If None, return all neighbors (within radius if specified)
        method: Backend to use - "faiss", "pytorch3d", "torch", or "scipy"
        return_distances: If True, return (distances, indices). If False, return only indices
        radius: Optional radius for radius search (only supported by some methods)
        
    Returns:
        If return_distances=True: (distances [N, k], indices [N, k])
        If return_distances=False: indices [N, k]
        
    Notes:
        - If k is None and radius is None: return 1 nearest neighbor
        - If k is provided: return k nearest neighbors (inf distance and -1 index for no match)
        - If radius is provided: search only within radius
        - If both k and radius: search within radius and limit to k results
    """
    # Use check_pc_xyz for point cloud validation
    check_pc_xyz(query_points)
    check_pc_xyz(reference_points)
    
    assert query_points.dtype == reference_points.dtype, f"dtype mismatch: {query_points.dtype} vs {reference_points.dtype}"
    assert query_points.device == reference_points.device, f"device mismatch: {query_points.device} vs {reference_points.device}"
    assert k is None or isinstance(k, int), f"k must be None or int, got {type(k)}"
    assert k is None or k > 0, f"k must be positive if provided, got {k}"
    assert method in ["faiss", "pytorch3d", "torch", "scipy"], f"Unknown method: {method}"
    assert isinstance(return_distances, bool), f"return_distances must be bool, got {type(return_distances)}"
    assert radius is None or isinstance(radius, (int, float)), f"radius must be None or numeric, got {type(radius)}"
    assert radius is None or radius > 0, f"radius must be positive if provided, got {radius}"
    
    # Default k=1 if neither k nor radius specified
    if k is None and radius is None:
        k = 1
    # If only radius specified, use all reference points as upper bound for k
    elif k is None and radius is not None:
        k = reference_points.shape[0]
    
    device = query_points.device
    
    if method == "faiss":
        distances, indices = _knn_faiss(
            query_points=query_points,
            reference_points=reference_points,
            k=k,
            radius=radius
        )
    elif method == "pytorch3d":
        distances, indices = _knn_pytorch3d(
            query_points=query_points,
            reference_points=reference_points,
            k=k,
            radius=radius
        )
    elif method == "torch":
        distances, indices = _knn_torch(
            query_points=query_points,
            reference_points=reference_points,
            k=k,
            radius=radius
        )
    elif method == "scipy":
        distances, indices = _knn_scipy(
            query_points=query_points,
            reference_points=reference_points,
            k=k,
            radius=radius
        )
    
    # Convert numpy to torch if needed
    if not isinstance(distances, torch.Tensor):
        assert isinstance(distances, np.ndarray), f"Expected np.ndarray, got {type(distances)}"
        distances = torch.from_numpy(distances).to(device)
    if not isinstance(indices, torch.Tensor):
        assert isinstance(indices, np.ndarray), f"Expected np.ndarray, got {type(indices)}"
        indices = torch.from_numpy(indices).to(device)
    
    # Ensure correct types
    assert distances.dtype in [torch.float32, torch.float64], f"Distances have wrong dtype: {distances.dtype}"
    indices = indices.long()
    
    assert distances.shape == (query_points.shape[0], k), f"Wrong distances shape: {distances.shape}"
    assert indices.shape == (query_points.shape[0], k), f"Wrong indices shape: {indices.shape}"
    
    if return_distances:
        return distances, indices
    else:
        return indices


def _knn_faiss(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: int = 1,
    radius: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """FAISS backend - supports both CPU and GPU, very fast for large datasets."""
    import faiss
    
    assert isinstance(query_points, torch.Tensor), "query_points must be torch.Tensor"
    assert isinstance(reference_points, torch.Tensor), "reference_points must be torch.Tensor"
    assert k > 0, f"k must be positive, got {k}"
    
    # Convert to numpy and ensure contiguous float32
    query_np = query_points.detach().cpu().numpy().astype(np.float32)
    reference_np = reference_points.detach().cpu().numpy().astype(np.float32)
    
    # Ensure contiguous arrays
    if not query_np.flags['C_CONTIGUOUS']:
        query_np = np.ascontiguousarray(query_np)
    if not reference_np.flags['C_CONTIGUOUS']:
        reference_np = np.ascontiguousarray(reference_np)
    
    assert query_np.dtype == np.float32, f"query_np must be float32, got {query_np.dtype}"
    assert reference_np.dtype == np.float32, f"reference_np must be float32, got {reference_np.dtype}"
    
    d = reference_np.shape[1]  # dimension
    assert d == 3, f"Dimension must be 3, got {d}"
    
    # Build index
    if query_points.is_cuda and hasattr(faiss, 'StandardGpuResources'):
        # Use GPU if available
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
    else:
        # Use CPU
        index = faiss.IndexFlatL2(d)
    
    # Add reference points to index
    index.add(reference_np)
    assert index.ntotal == len(reference_np), f"Index size mismatch: {index.ntotal} vs {len(reference_np)}"
    
    if radius is not None:
        assert isinstance(radius, (int, float)), f"radius must be numeric, got {type(radius)}"
        assert radius > 0, f"radius must be positive, got {radius}"
        
        # Range search within radius
        lims, distances, indices = index.range_search(query_np, radius * radius)  # FAISS uses squared distances
        assert len(lims) == len(query_np) + 1, f"lims length mismatch: {len(lims)} vs {len(query_np) + 1}"
        
        # Convert to k-nearest format (take first k within radius)
        n_queries = len(query_np)
        dist_out = np.full((n_queries, k), np.inf, dtype=np.float32)
        idx_out = np.full((n_queries, k), -1, dtype=np.int64)
        
        for i in range(n_queries):
            start, end = lims[i], lims[i + 1]
            assert start >= 0 and end >= start, f"Invalid lims: start={start}, end={end}"
            n_found = min(end - start, k)
            if n_found > 0:
                # Sort by distance and take k nearest
                range_dists = distances[start:end]
                range_indices = indices[start:end]
                assert len(range_dists) == n_found or len(range_dists) > k, f"Not enough distances: {len(range_dists)}"
                perm = np.argsort(range_dists)[:n_found]
                dist_out[i, :n_found] = np.sqrt(range_dists[perm])  # Convert back from squared
                idx_out[i, :n_found] = range_indices[perm]
        
        return dist_out, idx_out
    else:
        # Standard k-NN search  
        k_actual = min(k, len(reference_np))
        distances, indices = index.search(query_np, k_actual)
        assert distances.shape == (len(query_np), k_actual), f"Wrong distances shape: {distances.shape}"
        assert indices.shape == (len(query_np), k_actual), f"Wrong indices shape: {indices.shape}"
        distances = np.sqrt(distances)  # FAISS returns squared distances
        
        # If k > actual reference points, pad with inf/-1
        if k > k_actual:
            n_queries = len(query_np)
            dist_out = np.full((n_queries, k), np.inf, dtype=np.float32)
            idx_out = np.full((n_queries, k), -1, dtype=np.int64)
            
            dist_out[:, :k_actual] = distances
            idx_out[:, :k_actual] = indices
            
            return dist_out, idx_out
        else:
            return distances, indices


def _knn_pytorch3d(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: int = 1,
    radius: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch3D backend - GPU accelerated, exact results."""
    from pytorch3d.ops import knn_points
    
    assert isinstance(query_points, torch.Tensor), "query_points must be torch.Tensor"
    assert isinstance(reference_points, torch.Tensor), "reference_points must be torch.Tensor"
    assert k > 0, f"k must be positive, got {k}"
    assert query_points.shape[1] == 3, f"query_points must have 3 coordinates"
    assert reference_points.shape[1] == 3, f"reference_points must have 3 coordinates"
    
    # Handle k > reference points
    k_actual = min(k, reference_points.shape[0])
    
    # PyTorch3D expects batch dimension
    query_batch = query_points.unsqueeze(0)  # [1, N, D]
    reference_batch = reference_points.unsqueeze(0)  # [1, M, D]
    
    assert query_batch.shape[0] == 1, f"Batch dimension must be 1, got {query_batch.shape[0]}"
    assert reference_batch.shape[0] == 1, f"Batch dimension must be 1, got {reference_batch.shape[0]}"
    
    # Compute k-NN
    result = knn_points(query_batch, reference_batch, K=k_actual, return_nn=False)
    
    assert hasattr(result, 'dists'), "Result must have dists attribute"
    assert hasattr(result, 'idx'), "Result must have idx attribute"
    
    # Extract distances and indices, remove batch dimension
    distances = result.dists.squeeze(0).sqrt()  # PyTorch3D returns squared distances
    indices = result.idx.squeeze(0)
    
    assert distances.shape == (query_points.shape[0], k_actual), f"Wrong distances shape: {distances.shape}"
    assert indices.shape == (query_points.shape[0], k_actual), f"Wrong indices shape: {indices.shape}"
    
    # Apply radius filtering if specified
    if radius is not None:
        assert isinstance(radius, (int, float)), f"radius must be numeric, got {type(radius)}"
        assert radius > 0, f"radius must be positive, got {radius}"
        
        # Mask out points beyond radius
        radius_mask = distances > radius
        distances = distances.masked_fill(radius_mask, float('inf'))
        indices = indices.masked_fill(radius_mask, -1)
    
    # If k > actual reference points, pad with inf/-1
    if k > k_actual:
        n_queries = query_points.shape[0]
        dist_out = torch.full((n_queries, k), float('inf'), dtype=query_points.dtype, device=query_points.device)
        idx_out = torch.full((n_queries, k), -1, dtype=torch.long, device=query_points.device)
        
        dist_out[:, :k_actual] = distances
        idx_out[:, :k_actual] = indices
        
        return dist_out, idx_out
    else:
        return distances, indices


def _knn_torch(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: int = 1,
    radius: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch backend - memory intensive but works everywhere."""
    assert isinstance(query_points, torch.Tensor), "query_points must be torch.Tensor"
    assert isinstance(reference_points, torch.Tensor), "reference_points must be torch.Tensor"
    assert k > 0, f"k must be positive, got {k}"
    assert query_points.shape[1] == 3, f"query_points must have 3 coordinates"
    assert reference_points.shape[1] == 3, f"reference_points must have 3 coordinates"
    
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
    
    # Apply radius filter if specified
    if radius is not None:
        assert isinstance(radius, (int, float)), f"radius must be numeric, got {type(radius)}"
        assert radius > 0, f"radius must be positive, got {radius}"
        
        # Mask out points beyond radius (set to inf)
        radius_mask = distances_squared > (radius * radius)
        distances_squared = distances_squared.masked_fill(radius_mask, float('inf'))
    
    # Handle case where k > number of reference points
    k_actual = min(k, reference_points.shape[0])
    
    # Find k nearest neighbors
    if k_actual < reference_points.shape[0] // 10:
        # Use topk for small k (more efficient)
        dist_k, idx_k = distances_squared.topk(k_actual, dim=1, largest=False, sorted=True)
    else:
        # Use sort for large k
        sorted_dists, sorted_indices = distances_squared.sort(dim=1)
        dist_k = sorted_dists[:, :k_actual]
        idx_k = sorted_indices[:, :k_actual]
    
    # If k > actual results, pad with inf/-1
    if k > k_actual:
        n_queries = query_points.shape[0]
        dist_out = torch.full((n_queries, k), float('inf'), dtype=query_points.dtype, device=query_points.device)
        idx_out = torch.full((n_queries, k), -1, dtype=torch.long, device=query_points.device)
        
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


def _knn_scipy(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: int = 1,
    radius: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """SciPy backend using cKDTree - good for CPU, supports radius search."""
    from scipy.spatial import cKDTree
    
    assert isinstance(query_points, torch.Tensor), "query_points must be torch.Tensor"
    assert isinstance(reference_points, torch.Tensor), "reference_points must be torch.Tensor"
    assert k > 0, f"k must be positive, got {k}"
    assert query_points.shape[1] == 3, f"query_points must have 3 coordinates"
    assert reference_points.shape[1] == 3, f"reference_points must have 3 coordinates"
    
    # Convert to numpy
    query_np = query_points.detach().cpu().numpy()
    reference_np = reference_points.detach().cpu().numpy()
    
    assert query_np.shape == query_points.shape, f"Shape mismatch after conversion"
    assert reference_np.shape == reference_points.shape, f"Shape mismatch after conversion"
    
    # Build KD-tree
    tree = cKDTree(reference_np)
    assert tree.n == len(reference_np), f"Tree size mismatch: {tree.n} vs {len(reference_np)}"
    assert tree.m == 3, f"Tree dimension must be 3, got {tree.m}"
    
    if radius is not None:
        assert isinstance(radius, (int, float)), f"radius must be numeric, got {type(radius)}"
        assert radius > 0, f"radius must be positive, got {radius}"
        
        # Radius search
        # First get all points within radius
        indices_list = tree.query_ball_point(query_np, radius)
        assert len(indices_list) == len(query_np), f"indices_list length mismatch"
        
        # Convert to k-nearest format
        n_queries = len(query_np)
        dist_out = np.full((n_queries, k), np.inf, dtype=query_np.dtype)
        idx_out = np.full((n_queries, k), -1, dtype=np.int64)
        
        for i, neighbors in enumerate(indices_list):
            if len(neighbors) > 0:
                neighbors = np.array(neighbors, dtype=np.int64)
                assert neighbors.max() < len(reference_np), f"Invalid neighbor index: {neighbors.max()}"
                assert neighbors.min() >= 0, f"Invalid neighbor index: {neighbors.min()}"
                
                # Compute distances to all neighbors
                dists = np.linalg.norm(reference_np[neighbors] - query_np[i], axis=1)
                assert len(dists) == len(neighbors), f"Distance array length mismatch"
                assert (dists >= 0).all(), "Distances must be non-negative"
                
                # Sort and take k nearest
                n_found = min(len(neighbors), k)
                sorted_idx = np.argsort(dists)[:n_found]
                
                dist_out[i, :n_found] = dists[sorted_idx]
                idx_out[i, :n_found] = neighbors[sorted_idx]
        
        assert dist_out.shape == (n_queries, k), f"Wrong dist_out shape: {dist_out.shape}"
        assert idx_out.shape == (n_queries, k), f"Wrong idx_out shape: {idx_out.shape}"
        
        return dist_out, idx_out
    else:
        # Standard k-NN search
        k_actual = min(k, len(reference_np))
        distances, indices = tree.query(query_np, k=k_actual)
        
        # Handle case where k=1 (scipy returns 1D arrays)
        if k_actual == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        
        assert distances.shape == (len(query_np), k_actual), f"Wrong distances shape: {distances.shape}"
        assert indices.shape == (len(query_np), k_actual), f"Wrong indices shape: {indices.shape}"
        assert (distances >= 0).all(), "Distances must be non-negative"
        
        # If k > actual reference points, pad with inf/-1
        if k > k_actual:
            n_queries = len(query_np)
            dist_out = np.full((n_queries, k), np.inf, dtype=query_np.dtype)
            idx_out = np.full((n_queries, k), -1, dtype=np.int64)
            
            dist_out[:, :k_actual] = distances
            idx_out[:, :k_actual] = indices
            
            return dist_out, idx_out
        else:
            return distances, indices
