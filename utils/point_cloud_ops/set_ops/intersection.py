from typing import Tuple, Optional
from itertools import product
from scipy.spatial import cKDTree
import numpy as np
import torch


def _calculate_chunk_factor(src_points: torch.Tensor, tgt_points: torch.Tensor) -> int:
    """
    Calculate the optimal chunk factor based on available CUDA memory and point cloud size.

    Args:
        src_points: Source point cloud positions, shape (N, 3)
        tgt_points: Target point cloud positions, shape (M, 3)

    Returns:
        Chunk factor to use for recursive implementation
    """
    assert src_points.device == tgt_points.device

    if src_points.device.type == 'cpu':
        return 1

    # Get available CUDA memory in bytes
    available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()

    # Estimate memory needed for the full computation
    # Each point is 3 floats (12 bytes), and we need to compute N*M distances
    src_size = src_points.shape[0]
    tgt_size = tgt_points.shape[0]

    # Memory for source and target points
    points_memory = (src_size + tgt_size) * 3 * 4  # 4 bytes per float

    # Memory for distance matrix (N*M floats)
    distance_memory = src_size * tgt_size * 4

    # Memory for boolean matrix (N*M booleans, 1 byte each)
    boolean_memory = src_size * tgt_size

    # Total estimated memory needed
    total_memory_needed = points_memory + distance_memory + boolean_memory

    # If we need more memory than available, calculate chunk factor
    if total_memory_needed > available_memory:
        # Calculate how many times we need to divide the problem
        # We want to use at most 80% of available memory
        memory_ratio = total_memory_needed / (available_memory * 0.8)

        # Since we divide both dimensions, the chunk factor grows quadratically
        # with the number of divisions needed
        chunk_factor = int(np.ceil(np.sqrt(memory_ratio)))

        # Ensure chunk factor is at least 1
        return max(1, chunk_factor)

    # If we have enough memory, no need to chunk
    return 1


def _tensor_intersection(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the intersection between two point clouds using pure tensor operations.

    Args:
        src_points: Source point cloud positions, shape (N, 3)
        tgt_points: Target point cloud positions, shape (M, 3)
        radius: Distance radius for considering points as overlapping

    Returns:
        A tuple containing:
        - Indices of source points that are close to any target point
        - Indices of target points that are close to any source point
    """
    assert isinstance(src_points, torch.Tensor)
    assert isinstance(tgt_points, torch.Tensor)
    assert src_points.ndim == 2 and tgt_points.ndim == 2
    assert src_points.shape[1] == 3 and tgt_points.shape[1] == 3
    assert src_points.device == tgt_points.device

    # Reshape for broadcasting: (N, 1, 3) - (1, M, 3) = (N, M, 3)
    src_expanded = src_points.unsqueeze(1)  # Shape: (N, 1, 3)
    tgt_expanded = tgt_points.unsqueeze(0)  # Shape: (1, M, 3)

    # Calculate all pairwise distances: (N, M)
    distances = torch.norm(src_expanded - tgt_expanded, dim=2)

    # Find points within radius
    within_radius = distances < radius

    # Check if any target point is within radius of each source point
    src_overlapping = torch.any(within_radius, dim=1)
    src_overlapping_indices = torch.where(src_overlapping)[0]

    # Check if any source point is within radius of each target point
    tgt_overlapping = torch.any(within_radius, dim=0)
    tgt_overlapping_indices = torch.where(tgt_overlapping)[0]

    return src_overlapping_indices, tgt_overlapping_indices


def _tensor_intersection_recursive(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float,
    chunk_factor: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the intersection between two point clouds using a recursive divide-and-conquer approach
    to handle CUDA out-of-memory issues. This implementation symmetrically divides both source and
    target point clouds when OOM occurs.

    Args:
        src_points: Source point cloud positions, shape (N, 3)
        tgt_points: Target point cloud positions, shape (M, 3)
        radius: Distance radius for considering points as overlapping
        chunk_factor: Factor to divide the point clouds by (increases with recursion)

    Returns:
        A tuple containing:
        - Indices of source points that are close to any target point
        - Indices of target points that are close to any source point
    """
    assert src_points.device == tgt_points.device

    # If chunk_factor is not provided, calculate it based on available memory
    if chunk_factor is None:
        chunk_factor = _calculate_chunk_factor(src_points, tgt_points)
        if chunk_factor > 1:
            print(f"Pre-calculated initial chunk factor: {chunk_factor}")

    try:
        # Try to compute the intersection with the current chunk size
        return _tensor_intersection(src_points, tgt_points, radius)
    except torch.cuda.OutOfMemoryError:
        # If OOM occurs, divide the problem and recursively solve
        print(f"CUDA OOM with chunk_factor={chunk_factor}, dividing problem symmetrically...")

        # Divide both source and target points into chunks
        num_chunks = 2 * chunk_factor
        src_chunks = list(torch.chunk(src_points, num_chunks))
        tgt_chunks = list(torch.chunk(tgt_points, num_chunks))

        # Initialize result tensors
        src_overlapping_indices_list = []
        tgt_overlapping_indices_list = []

        # Calculate chunk sizes and starting indices
        src_chunk_sizes = [len(chunk) for chunk in src_chunks]
        tgt_chunk_sizes = [len(chunk) for chunk in tgt_chunks]

        src_start_indices = [sum(src_chunk_sizes[:i]) for i in range(len(src_chunks))]
        tgt_start_indices = [sum(tgt_chunk_sizes[:i]) for i in range(len(tgt_chunks))]

        # Process each pair of source and target chunks using itertools.product
        for (i, src_chunk), (j, tgt_chunk) in product(enumerate(src_chunks), enumerate(tgt_chunks)):
            # Recursively process this pair of chunks with a larger chunk factor
            src_indices, tgt_indices = _tensor_intersection_recursive(
                src_chunk, tgt_chunk, radius, chunk_factor * 2
            )

            # Adjust source indices to account for chunking
            if len(src_indices) > 0:
                adjusted_src_indices = src_indices + src_start_indices[i]
                src_overlapping_indices_list.append(adjusted_src_indices)

            # Adjust target indices to account for chunking
            if len(tgt_indices) > 0:
                adjusted_tgt_indices = tgt_indices + tgt_start_indices[j]
                tgt_overlapping_indices_list.append(adjusted_tgt_indices)

        # Combine results
        if src_overlapping_indices_list:
            src_overlapping_indices = torch.unique(torch.cat(src_overlapping_indices_list))
        else:
            src_overlapping_indices = torch.tensor([], dtype=torch.long, device=src_points.device)

        if tgt_overlapping_indices_list:
            tgt_overlapping_indices = torch.unique(torch.cat(tgt_overlapping_indices_list))
        else:
            tgt_overlapping_indices = torch.tensor([], dtype=torch.long, device=tgt_points.device)

        return src_overlapping_indices, tgt_overlapping_indices


def _kdtree_intersection(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the intersection between two point clouds using KD-tree.

    Args:
        src_points: Source point cloud positions, shape (N, 3)
        tgt_points: Target point cloud positions, shape (M, 3)
        radius: Distance radius for considering points as overlapping

    Returns:
        A tuple containing:
        - Indices of source points that are close to any target point
        - Indices of target points that are close to any source point
    """
    assert isinstance(src_points, torch.Tensor)
    assert isinstance(tgt_points, torch.Tensor)
    assert src_points.ndim == 2 and tgt_points.ndim == 2
    assert src_points.shape[1] == 3 and tgt_points.shape[1] == 3

    # Convert to numpy for KD-tree operations
    src_np = src_points.cpu().numpy()
    tgt_np = tgt_points.cpu().numpy()

    # Build KD-tree for target points
    tgt_tree = cKDTree(tgt_np)

    # Find source points that are close to any target point
    # Query with radius returns all points within radius
    src_overlapping_indices = []
    for i, src_point in enumerate(src_np):
        # Find all target points within radius of this source point
        neighbors = tgt_tree.query_ball_point(src_point, radius)
        if len(neighbors) > 0:
            src_overlapping_indices.append(i)

    # Build KD-tree for source points
    src_tree = cKDTree(src_np)

    # Find target points that are close to any source point
    tgt_overlapping_indices = []
    for i, tgt_point in enumerate(tgt_np):
        # Find all source points within radius of this target point
        neighbors = src_tree.query_ball_point(tgt_point, radius)
        if len(neighbors) > 0:
            tgt_overlapping_indices.append(i)

    # Convert lists to tensors
    src_overlapping_indices = torch.tensor(src_overlapping_indices, device=src_points.device)
    tgt_overlapping_indices = torch.tensor(tgt_overlapping_indices, device=tgt_points.device)

    return src_overlapping_indices, tgt_overlapping_indices


# Set pc_intersection to use the _tensor_intersection_recursive implementation
# since it was the winning method in the benchmark
pc_intersection = _tensor_intersection_recursive


def compute_pc_iou(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float,
) -> float:
    """
    Calculate the overlap ratio (IoU) between two point clouds.

    Args:
        src_points: Source point cloud positions, shape (N, 3)
        tgt_points: Target point cloud positions, shape (M, 3)
        radius: Distance radius for considering points as overlapping

    Returns:
        The overlap ratio, defined as the number of overlapping points divided by the total number of points
    """
    assert isinstance(src_points, torch.Tensor)
    assert isinstance(tgt_points, torch.Tensor)
    assert src_points.ndim == 2 and tgt_points.ndim == 2
    assert src_points.shape[1] == 3 and tgt_points.shape[1] == 3
    # Get overlapping indices
    src_overlapping_indices, tgt_overlapping_indices = pc_intersection(
        src_points, tgt_points, radius
    )

    # Calculate total overlapping points (union of both sets)
    total_overlapping = len(src_overlapping_indices) + len(tgt_overlapping_indices)
    total_points = len(src_points) + len(tgt_points)

    # Calculate overlap ratio
    overlap_ratio = total_overlapping / total_points if total_points > 0 else 0

    return overlap_ratio
