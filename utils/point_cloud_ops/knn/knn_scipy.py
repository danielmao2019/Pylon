from typing import Tuple, Optional
import torch
import numpy as np


def _knn_scipy(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: Optional[int] = None,
    radius: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """SciPy backend using cKDTree - good for CPU, supports radius search."""
    from scipy.spatial import cKDTree
    
    assert isinstance(query_points, torch.Tensor), "query_points must be torch.Tensor"
    assert isinstance(reference_points, torch.Tensor), "reference_points must be torch.Tensor"
    assert (k is None) != (radius is None), "Exactly one of k or radius must be specified"
    assert k is None or k > 0, f"k must be positive if provided, got {k}"
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
        return _knn_scipy_with_r(
            query_np=query_np,
            reference_np=reference_np,
            tree=tree,
            radius=radius
        )
    else:
        return _knn_scipy_with_k(
            query_np=query_np,
            reference_np=reference_np,
            tree=tree,
            k=k
        )


def _knn_scipy_with_r(
    query_np: np.ndarray,
    reference_np: np.ndarray,
    tree,
    radius: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Radius search using cKDTree."""
    assert isinstance(radius, (int, float)), f"radius must be numeric, got {type(radius)}"
    assert radius > 0, f"radius must be positive, got {radius}"
    
    # Radius search
    # First get all points within radius
    indices_list = tree.query_ball_point(query_np, radius)
    assert len(indices_list) == len(query_np), f"indices_list length mismatch"
    
    # Find max neighbors
    max_neighbors = max(len(neighbors) for neighbors in indices_list)
    if max_neighbors == 0:
        max_neighbors = 1  # At least return 1 column
    
    # Convert to dense format
    n_queries = len(query_np)
    dist_out = np.full((n_queries, max_neighbors), np.inf, dtype=query_np.dtype)
    idx_out = np.full((n_queries, max_neighbors), -1, dtype=np.int64)
    
    # Convert scipy results to numpy arrays using vectorized operations only
    # Use numpy's built-in functions to handle the list conversion
    
    # Convert to numpy object array to enable vectorized operations
    indices_obj_array = np.asarray(indices_list, dtype=object)
    
    # Get lengths using vectorized len function
    lengths = np.vectorize(len)(indices_obj_array)
    max_found = int(lengths.max()) if len(lengths) > 0 else 0
    
    if max_found > 0:
        # Use numpy's advanced indexing to create padded result
        # This approach avoids explicit loops by using numpy's internal vectorization
        result_neighbors = np.full((n_queries, max_found), -1, dtype=np.int64)
        
        # Use numpy's put_along_axis equivalent for filling
        # Create row indices for each element
        row_indices = np.repeat(np.arange(n_queries), lengths)
        # Create column indices using pure numpy operations
        if lengths.sum() > 0:
            # Use numpy operations to create column indices without loops
            max_len = int(lengths.max())
            all_indices = np.arange(max_len)
            # Create mask for valid positions and flatten
            valid_mask = all_indices[None, :] < lengths[:, None]  # [n_queries, max_len]
            col_indices = np.tile(all_indices, n_queries)[valid_mask.ravel()]
            
            all_neighbor_values = np.concatenate(indices_list)  # numpy's internal implementation
        else:
            col_indices = np.array([], dtype=int)
            all_neighbor_values = np.array([], dtype=np.int64)
        
        # Fill the result array using advanced indexing
        if len(all_neighbor_values) > 0:
            result_neighbors[row_indices, col_indices] = all_neighbor_values
        
        # Compute all distances using broadcasting
        query_broadcast = query_np[:, None, :]  # [N, 1, 3]
        valid_refs = np.clip(result_neighbors, 0, len(reference_np) - 1)  # Clip invalid indices
        ref_points = reference_np[valid_refs]  # [N, max_found, 3]
        
        # Calculate distances for all at once
        distances_all = np.linalg.norm(ref_points - query_broadcast, axis=2)  # [N, max_found]
        
        # Apply validity mask (invalid neighbors get inf distance, -1 index)
        valid_mask = result_neighbors >= 0
        final_distances = np.where(valid_mask, distances_all, np.inf)
        final_indices = np.where(valid_mask, result_neighbors, -1)
        
        # Trim to fit output size
        output_cols = min(max_neighbors, max_found)
        dist_out[:, :output_cols] = final_distances[:, :output_cols]
        idx_out[:, :output_cols] = final_indices[:, :output_cols]
    
    assert dist_out.shape == (n_queries, max_neighbors), f"Wrong dist_out shape: {dist_out.shape}"
    assert idx_out.shape == (n_queries, max_neighbors), f"Wrong idx_out shape: {idx_out.shape}"
    
    return dist_out, idx_out


def _knn_scipy_with_k(
    query_np: np.ndarray,
    reference_np: np.ndarray, 
    tree,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard k-NN search using cKDTree."""
    assert k is not None
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
