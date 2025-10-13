from typing import Tuple, Optional
import torch
from pytorch3d.ops import knn_points
import logging
import time

# Configure logging to show timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def _knn_pytorch3d(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: Optional[int] = None,
    radius: Optional[float] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logging.info("Calling _knn_pytorch3d...")
    """PyTorch3D backend - GPU accelerated, exact results."""
    assert isinstance(query_points, torch.Tensor), "query_points must be torch.Tensor"
    assert isinstance(
        reference_points, torch.Tensor
    ), "reference_points must be torch.Tensor"
    assert (k is None) != (
        radius is None
    ), "Exactly one of k or radius must be specified"
    assert k is None or k > 0, f"k must be positive if provided, got {k}"
    assert query_points.shape[1] == 3, f"query_points must have 3 coordinates"
    assert reference_points.shape[1] == 3, f"reference_points must have 3 coordinates"
    assert (
        chunk_size is None or chunk_size > 0
    ), f"chunk_size must be positive if provided, got {chunk_size}"

    n_queries = query_points.shape[0]

    # Process in chunks if chunk_size is provided and we have more queries than chunk_size
    if chunk_size is not None and n_queries > chunk_size:
        distances_list = []
        indices_list = []
        n_chunks = (n_queries + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_queries)

            # Track time for this chunk
            chunk_start_time = time.time()

            # Log progress with timestamp
            logging.info(
                f"Processing chunk {chunk_idx + 1}/{n_chunks} "
                f"(queries {start_idx}:{end_idx})"
            )

            chunk_queries = query_points[start_idx:end_idx]

            # Process chunk
            if radius is not None:
                chunk_distances, chunk_indices = _knn_pytorch3d_with_r(
                    query_points=chunk_queries,
                    reference_points=reference_points,
                    radius=radius,
                )
            else:
                assert k is not None, "k must not be None for k-NN search"
                chunk_distances, chunk_indices = _knn_pytorch3d_with_k(
                    query_points=chunk_queries, reference_points=reference_points, k=k
                )

            distances_list.append(chunk_distances)
            indices_list.append(chunk_indices)

            # Log time taken for this chunk
            chunk_time = time.time() - chunk_start_time
            logging.info(
                f"Chunk {chunk_idx + 1}/{n_chunks} completed in {chunk_time:.3f} seconds"
            )

        # For radius search, we need to pad chunks to the same width before concatenating
        if radius is not None:
            # Find the maximum width across all chunks
            max_width = max(chunk.shape[1] for chunk in distances_list)

            # Pad all chunks to the same width
            padded_distances = []
            padded_indices = []

            for dist_chunk, idx_chunk in zip(distances_list, indices_list):
                current_width = dist_chunk.shape[1]
                if current_width < max_width:
                    # Pad with inf for distances and -1 for indices
                    dist_pad = torch.full(
                        (dist_chunk.shape[0], max_width - current_width),
                        float('inf'),
                        dtype=dist_chunk.dtype,
                        device=dist_chunk.device,
                    )
                    idx_pad = torch.full(
                        (idx_chunk.shape[0], max_width - current_width),
                        -1,
                        dtype=idx_chunk.dtype,
                        device=idx_chunk.device,
                    )
                    dist_chunk = torch.cat([dist_chunk, dist_pad], dim=1)
                    idx_chunk = torch.cat([idx_chunk, idx_pad], dim=1)

                padded_distances.append(dist_chunk)
                padded_indices.append(idx_chunk)

            return torch.cat(padded_distances, dim=0), torch.cat(padded_indices, dim=0)
        else:
            # For k-NN, all chunks have the same width
            return torch.cat(distances_list, dim=0), torch.cat(indices_list, dim=0)

    # Process all queries at once (either no chunking or small dataset)
    if radius is not None:
        return _knn_pytorch3d_with_r(
            query_points=query_points, reference_points=reference_points, radius=radius
        )
    else:
        assert k is not None, "k must not be None for k-NN search"
        return _knn_pytorch3d_with_k(
            query_points=query_points, reference_points=reference_points, k=k
        )


def _knn_pytorch3d_with_k(
    query_points: torch.Tensor, reference_points: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch3D k-NN search implementation."""
    assert k > 0, f"k must be positive, got {k}"

    # Handle k > reference points
    k_actual = min(k, reference_points.shape[0])

    # PyTorch3D expects batch dimension
    query_batch = query_points.unsqueeze(0)  # [1, N, D]
    reference_batch = reference_points.unsqueeze(0)  # [1, M, D]

    assert (
        query_batch.shape[0] == 1
    ), f"Batch dimension must be 1, got {query_batch.shape[0]}"
    assert (
        reference_batch.shape[0] == 1
    ), f"Batch dimension must be 1, got {reference_batch.shape[0]}"

    # Compute k-NN
    original_device = query_points.device
    result = knn_points(
        query_batch.cuda(), reference_batch.cuda(), K=k_actual, return_nn=False
    )

    assert hasattr(result, 'dists'), "Result must have dists attribute"
    assert hasattr(result, 'idx'), "Result must have idx attribute"

    # Extract distances and indices, remove batch dimension
    distances = (
        result.dists.squeeze(0).sqrt().to(original_device)
    )  # PyTorch3D returns squared distances
    indices = result.idx.squeeze(0).to(original_device)

    assert distances.shape == (
        query_points.shape[0],
        k_actual,
    ), f"Wrong distances shape: {distances.shape}"
    assert indices.shape == (
        query_points.shape[0],
        k_actual,
    ), f"Wrong indices shape: {indices.shape}"

    # If k > actual reference points, pad with inf/-1
    if k > k_actual:
        n_queries = query_points.shape[0]
        dist_out = torch.full(
            (n_queries, k),
            float('inf'),
            dtype=query_points.dtype,
            device=query_points.device,
        )
        idx_out = torch.full(
            (n_queries, k), -1, dtype=torch.long, device=query_points.device
        )

        dist_out[:, :k_actual] = distances
        idx_out[:, :k_actual] = indices

        return dist_out, idx_out
    else:
        return distances, indices


def _knn_pytorch3d_with_r(
    query_points: torch.Tensor, reference_points: torch.Tensor, radius: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch3D radius search implementation that preserves original order."""
    assert isinstance(
        radius, (int, float)
    ), f"radius must be numeric, got {type(radius)}"
    assert radius > 0, f"radius must be positive, got {radius}"

    # For radius search, do k-NN with k=all points, then filter and re-sort by original order
    k_actual = reference_points.shape[0]

    # PyTorch3D expects batch dimension
    query_batch = query_points.unsqueeze(0)  # [1, N, D]
    reference_batch = reference_points.unsqueeze(0)  # [1, M, D]

    assert (
        query_batch.shape[0] == 1
    ), f"Batch dimension must be 1, got {query_batch.shape[0]}"
    assert (
        reference_batch.shape[0] == 1
    ), f"Batch dimension must be 1, got {reference_batch.shape[0]}"

    # Do k-NN search with all points
    original_device = query_points.device
    result = knn_points(
        query_batch.cuda(), reference_batch.cuda(), K=k_actual, return_nn=False
    )

    assert hasattr(result, 'dists'), "Result must have dists attribute"
    assert hasattr(result, 'idx'), "Result must have idx attribute"

    distances = (
        result.dists.squeeze(0).sqrt().to(original_device)
    )  # PyTorch3D returns squared distances
    indices = result.idx.squeeze(0).to(original_device)

    # Mask out points beyond radius
    radius_mask = distances > radius
    distances = distances.masked_fill(radius_mask, float('inf'))
    indices = indices.masked_fill(radius_mask, -1)

    # Re-sort by original reference point order to preserve original ordering
    # Use argsort on indices to get the permutation that restores original order
    valid_mask = indices >= 0  # Valid neighbors (not -1)

    # For each query point, we need to sort valid indices back to original order
    # Create a sorting permutation based on the reference indices
    sort_indices = torch.argsort(
        torch.where(
            valid_mask, indices, torch.tensor(float('inf'), device=indices.device)
        ),
        dim=1,
    )

    # Apply the sorting permutation to both distances and indices
    distances_reordered = distances.gather(1, sort_indices)
    indices_reordered = indices.gather(1, sort_indices)

    # Find max valid neighbors and trim
    valid_per_query = (distances_reordered != float('inf')).sum(dim=1)
    max_neighbors = valid_per_query.max().item()
    if max_neighbors == 0:
        max_neighbors = 1  # At least return 1 column even if all inf

    return distances_reordered[:, :max_neighbors], indices_reordered[:, :max_neighbors]
