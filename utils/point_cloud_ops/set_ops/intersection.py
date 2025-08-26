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
    from utils.input_checks.point_cloud import check_pc_xyz
    
    # Validate point clouds using check_pc_xyz
    check_pc_xyz(src_points)
    check_pc_xyz(tgt_points)
    
    # Validate radius parameter
    assert isinstance(radius, (int, float)), f"radius must be int or float, got {type(radius)}"
    assert radius > 0, f"radius must be greater than 0, got {radius}"
    
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


def get_nearest_neighbor_distances(
    query_points: torch.Tensor,
    support_points: torch.Tensor,
) -> torch.Tensor:
    """Get nearest neighbor distances for query points in support points.
    
    This function replicates GeoTransformer's get_nearest_neighbor behavior.
    
    Args:
        query_points: Query point cloud positions, shape (N, 3)
        support_points: Support point cloud positions, shape (M, 3)
        
    Returns:
        Distances to nearest neighbors, shape (N,)
    """
    print(f"DEBUG: get_nearest_neighbor_distances() starting with query_points shape: {query_points.shape}, support_points shape: {support_points.shape}")
    
    from utils.point_cloud_ops.knn import knn
    from utils.input_checks.point_cloud import check_pc_xyz
    
    # Validate inputs
    check_pc_xyz(query_points)
    check_pc_xyz(support_points)
    
    print("DEBUG: get_nearest_neighbor_distances() using KNN to find nearest neighbors")
    
    # Find nearest neighbors using KNN module
    distances, _ = knn(
        query_points=query_points,
        reference_points=support_points,
        k=1,  # Find single nearest neighbor
        return_distances=True
    )
    
    print(f"DEBUG: get_nearest_neighbor_distances() KNN completed, distances shape: {distances.shape}")
    
    # Squeeze to get shape (N,) instead of (N, 1)
    distances = distances.squeeze(1)
    print(f"DEBUG: get_nearest_neighbor_distances() completed, final tensor shape: {distances.shape}")
    
    return distances


def compute_registration_overlap(
    ref_points: torch.Tensor,
    src_points: torch.Tensor,
    transform: Optional[torch.Tensor] = None,
    positive_radius: float = 0.1
) -> float:
    """Compute overlap between two point clouds (GeoTransformer style).
    
    This function replicates GeoTransformer's compute_overlap behavior:
    - Directional overlap: fraction of ref points with src neighbors within radius
    - Used for filtering registration pairs based on coverage
    
    Args:
        ref_points: Reference point cloud positions, shape (N, 3)
        src_points: Source point cloud positions, shape (M, 3)
        transform: Optional 4x4 transformation matrix to apply to src_points
        positive_radius: Distance threshold for considering points as overlapping
        
    Returns:
        Overlap ratio as fraction of reference points with close source neighbors
    """
    print(f"DEBUG: compute_registration_overlap() starting with ref_points shape: {ref_points.shape}, src_points shape: {src_points.shape}")
    print(f"DEBUG: compute_registration_overlap() positive_radius: {positive_radius}, transform provided: {transform is not None}")
    assert isinstance(ref_points, torch.Tensor)
    assert isinstance(src_points, torch.Tensor)
    assert ref_points.ndim == 2 and src_points.ndim == 2
    assert ref_points.shape[1] == 3 and src_points.shape[1] == 3
    assert src_points.dtype == ref_points.dtype
    
    # Apply transformation to source points if provided
    if transform is not None:
        print("DEBUG: compute_registration_overlap() applying transformation to source points")
        assert isinstance(transform, torch.Tensor)
        assert transform.shape == (4, 4)
        transform = transform.to(src_points.dtype)
        # Apply SE(3) transformation: src_transformed = (R @ src.T + t).T
        R = transform[:3, :3]
        t = transform[:3, 3]
        src_points = (R @ src_points.T + t.unsqueeze(1)).T
        print(f"DEBUG: compute_registration_overlap() transformation applied, new src_points shape: {src_points.shape}")
    
    # Get nearest neighbor distances (ref -> src)
    print("DEBUG: compute_registration_overlap() computing nearest neighbor distances")
    nn_distances = get_nearest_neighbor_distances(ref_points, src_points)
    print(f"DEBUG: compute_registration_overlap() nn_distances computed, shape: {nn_distances.shape}")
    
    # Compute overlap as fraction of ref points with neighbors within radius
    print("DEBUG: compute_registration_overlap() computing overlap fraction")
    overlap = torch.mean((nn_distances < positive_radius).float()).item()
    print(f"DEBUG: compute_registration_overlap() computed overlap: {overlap}")
    
    return overlap
