from typing import Any, Optional, Tuple, Union

import numpy as np
import torch


def _to_numpy_f32(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor or numpy array to contiguous float32 numpy array."""
    # Input validations
    assert isinstance(data, (torch.Tensor, np.ndarray)), f"{type(data)=}"

    if isinstance(data, torch.Tensor):
        if data.dtype != torch.float32:
            data = data.float()
        if not data.is_contiguous():
            data = data.contiguous()
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        return data

    assert False, f"Unhandled data type: {type(data)}"


def _knn_faiss(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: Optional[int] = None,
    radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """FAISS backend - multi-GPU for k-NN, single GPU for range search."""
    import faiss

    assert isinstance(query_points, torch.Tensor), "query_points must be torch.Tensor"
    assert isinstance(
        reference_points, torch.Tensor
    ), "reference_points must be torch.Tensor"
    assert (k is None) != (
        radius is None
    ), "Exactly one of k or radius must be specified"
    assert k is None or k > 0, f"k must be positive if provided, got {k}"
    assert (
        radius is None or radius > 0
    ), f"radius must be positive if provided, got {radius}"

    # Check GPU availability for acceleration
    assert hasattr(faiss, 'StandardGpuResources'), "FAISS GPU support not available"
    ngpus = faiss.get_num_gpus()
    assert ngpus > 0, "No CUDA GPUs detected by FAISS"

    d = reference_points.shape[1]  # dimension
    assert d == 3, f"Dimension must be 3, got {d}"

    # Convert inputs to contiguous float32 numpy arrays
    reference_np = _to_numpy_f32(reference_points)
    query_np = _to_numpy_f32(query_points)

    if radius is not None:
        return _knn_faiss_with_r(
            query_np=query_np,
            reference_np=reference_np,
            radius=radius,
            d=d,
            faiss=faiss,
        )
    else:
        assert k is not None, "k must not be None for k-NN search"
        return _knn_faiss_with_k(
            query_np=query_np,
            reference_np=reference_np,
            k=k,
            d=d,
            ngpus=ngpus,
            faiss=faiss,
        )


def _knn_faiss_with_k(
    query_np: np.ndarray,
    reference_np: np.ndarray,
    k: int,
    d: int,
    ngpus: int,
    faiss: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """FAISS backend for k-NN search - multi-GPU acceleration."""
    assert k > 0, f"k must be positive, got {k}"

    # k-NN search - use GPU acceleration
    if ngpus > 1:
        # Multi-GPU setup: create CPU index first, then transfer to all GPUs
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(reference_np)
        index = faiss.index_cpu_to_all_gpus(cpu_index)
    else:
        # Single GPU setup
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
        index.add(reference_np)

    assert (
        index.ntotal == reference_np.shape[0]
    ), f"Index size mismatch: {index.ntotal} vs {reference_np.shape[0]}"

    # Standard k-NN search
    k_actual = min(k, len(reference_np))
    distances, indices = index.search(query_np, k_actual)
    assert distances.shape == (
        len(query_np),
        k_actual,
    ), f"Wrong distances shape: {distances.shape}"
    assert indices.shape == (
        len(query_np),
        k_actual,
    ), f"Wrong indices shape: {indices.shape}"
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


def _knn_faiss_with_r(
    query_np: np.ndarray,
    reference_np: np.ndarray,
    radius: float,
    d: int,
    faiss: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """FAISS backend for range search - CPU-based."""
    assert isinstance(
        radius, (int, float)
    ), f"radius must be numeric, got {type(radius)}"
    assert radius > 0, f"radius must be positive, got {radius}"

    # Range search requires CPU index (GPU doesn't support range_search)
    index = faiss.IndexFlatL2(d)
    index.add(reference_np)

    assert (
        index.ntotal == reference_np.shape[0]
    ), f"Index size mismatch: {index.ntotal} vs {reference_np.shape[0]}"

    # Range search within radius (using CPU index)
    lims, distances, indices = index.range_search(
        query_np, radius * radius
    )  # FAISS uses squared distances
    assert (
        len(lims) == len(query_np) + 1
    ), f"lims length mismatch: {len(lims)} vs {len(query_np) + 1}"

    # Convert FAISS range search results to dense format
    return _faiss_range_search_to_dense(lims=lims, distances=distances, indices=indices)


def _faiss_range_search_to_dense(
    lims: np.ndarray, distances: np.ndarray, indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert FAISS range_search results to dense [N, k] format.

    Args:
        lims: Array of length n_queries + 1 indicating where each query's results start/end
        distances: Flat array of all distances (squared) for all queries concatenated
        indices: Flat array of all indices for all queries concatenated

    Returns:
        distances_dense: [N, k] array with inf padding, k = max neighbors found
        indices_dense: [N, k] array with -1 padding, k = max neighbors found
    """
    assert isinstance(lims, np.ndarray), f"lims must be np.ndarray, got {type(lims)}"
    assert lims.dtype == np.uint64, f"lims must be uint64, got {lims.dtype}"
    assert isinstance(
        distances, np.ndarray
    ), f"distances must be np.ndarray, got {type(distances)}"
    assert (
        distances.dtype == np.float32
    ), f"distances must be float32, got {distances.dtype}"
    assert isinstance(
        indices, np.ndarray
    ), f"indices must be np.ndarray, got {type(indices)}"
    assert indices.dtype == np.int64, f"indices must be int64, got {indices.dtype}"

    # Convert lims to int64 at the start to avoid casting issues
    lims = lims.astype(np.int64)

    n_queries = len(lims) - 1
    assert len(distances) == len(
        indices
    ), f"distances and indices must have same length"
    assert lims[-1] == len(distances), f"lims[-1] must equal len(distances)"

    # Find max number of neighbors for any query point
    counts = np.diff(lims)  # Now already int64
    k = int(np.max(counts)) if len(counts) > 0 else 0

    if k == 0:
        # No neighbors found for any query
        distances_dense = np.full((n_queries, 1), np.inf, dtype=np.float32)
        indices_dense = np.full((n_queries, 1), -1, dtype=np.int64)
        return distances_dense, indices_dense

    # Initialize output arrays
    distances_dense = np.full((n_queries, k), np.inf, dtype=np.float32)
    indices_dense = np.full((n_queries, k), -1, dtype=np.int64)

    if len(distances) == 0:
        return distances_dense, indices_dense

    # Convert squared distances to actual distances
    distances_sqrt = np.sqrt(distances)

    # Create query_id and within-query position arrays
    query_ids = np.repeat(np.arange(n_queries, dtype=np.int64), counts)

    # Create within-query positions
    global_positions = np.arange(len(distances), dtype=np.int64)
    query_starts = np.repeat(lims[:-1], counts)  # lims is already int64
    within_query_pos = global_positions - query_starts

    # Fill output arrays using advanced indexing
    distances_dense[query_ids, within_query_pos] = distances_sqrt
    indices_dense[query_ids, within_query_pos] = indices

    return distances_dense, indices_dense
