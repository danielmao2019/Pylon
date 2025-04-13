#!/usr/bin/env python
"""
Benchmark script for comparing different point cloud intersection implementations.
"""
import time
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
import sys
from functools import partial
from itertools import product

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the KD-tree implementation
from utils.point_cloud_ops.set_ops.intersection import pc_intersection as kdtree_intersection

# Assert GPU is available
assert torch.cuda.is_available(), "This benchmark requires a GPU to run"


def tensor_intersection(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the intersection between two point clouds using pure tensor operations.

    Args:
        src_points: Source point cloud positions, shape (N, 3)
        tgt_points: Target point cloud positions, shape (M, 3)
        radius: Distance radius for considering points as overlapping
        device: Device to run the computation on ('cpu' or 'cuda')

    Returns:
        A tuple containing:
        - Indices of source points that are close to any target point
        - Indices of target points that are close to any source point
    """
    assert isinstance(src_points, torch.Tensor)
    assert isinstance(tgt_points, torch.Tensor)
    assert src_points.ndim == 2 and tgt_points.ndim == 2
    assert src_points.shape[1] == 3 and tgt_points.shape[1] == 3

    # Move tensors to the specified device
    src_points = src_points.to(device)
    tgt_points = tgt_points.to(device)

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

    # Move results back to the original device
    return src_overlapping_indices, tgt_overlapping_indices


def tensor_intersection_recursive(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float,
    device: str = 'cuda',
    chunk_factor: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the intersection between two point clouds using a recursive divide-and-conquer approach
    to handle CUDA out-of-memory issues. This implementation symmetrically divides both source and
    target point clouds when OOM occurs.

    Args:
        src_points: Source point cloud positions, shape (N, 3)
        tgt_points: Target point cloud positions, shape (M, 3)
        radius: Distance radius for considering points as overlapping
        device: Device to run the computation on ('cpu' or 'cuda')
        chunk_factor: Factor to divide the point clouds by (increases with recursion)

    Returns:
        A tuple containing:
        - Indices of source points that are close to any target point
        - Indices of target points that are close to any source point
    """
    try:
        # Try to compute the intersection with the current chunk size
        return tensor_intersection(src_points, tgt_points, radius, device)
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
            src_indices, tgt_indices = tensor_intersection_recursive(
                src_chunk, tgt_chunk, radius, device, chunk_factor * 2
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
            src_overlapping_indices = torch.tensor([], dtype=torch.long, device=device)

        if tgt_overlapping_indices_list:
            tgt_overlapping_indices = torch.unique(torch.cat(tgt_overlapping_indices_list))
        else:
            tgt_overlapping_indices = torch.tensor([], dtype=torch.long, device=device)

        return src_overlapping_indices, tgt_overlapping_indices


def generate_random_point_clouds(num_src: int, num_tgt: int, overlap_ratio: float = 0.3,
                                radius: float = 0.1, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random point clouds with controlled overlap.

    Args:
        num_src: Number of points in source point cloud
        num_tgt: Number of points in target point cloud
        overlap_ratio: Approximate ratio of points that should overlap
        radius: Distance radius for considering points as overlapping
        device: Device to place tensors on

    Returns:
        Tuple of (src_points, tgt_points)
    """
    # Generate source points in a unit cube
    src_points = torch.rand(num_src, 3, device=device)

    # Generate target points with controlled overlap
    num_overlap = int(min(num_src, num_tgt) * overlap_ratio)

    # Create overlapping points by adding small random offsets to some source points
    overlap_indices = torch.randperm(num_src)[:num_overlap]
    overlapping_points = src_points[overlap_indices] + torch.randn(num_overlap, 3, device=device) * radius * 0.5

    # Create non-overlapping points
    non_overlap_count = num_tgt - num_overlap
    non_overlapping_points = torch.rand(non_overlap_count, 3, device=device)

    # Combine overlapping and non-overlapping points
    tgt_points = torch.cat([overlapping_points, non_overlapping_points])

    # Shuffle target points
    tgt_indices = torch.randperm(num_tgt)
    tgt_points = tgt_points[tgt_indices]

    return src_points, tgt_points


def verify_results(reference_result: Tuple[torch.Tensor, torch.Tensor],
                  test_result: Tuple[torch.Tensor, torch.Tensor],
                  impl_name: str) -> bool:
    """
    Verify that two implementations produce the same results.

    Args:
        reference_result: Result from the reference implementation
        test_result: Result from the test implementation
        impl_name: Name of the test implementation

    Returns:
        True if results match, False otherwise
    """
    src_ref, tgt_ref = reference_result
    src_test, tgt_test = test_result

    # Check if the results have the same length
    if len(src_ref) != len(src_test) or len(tgt_ref) != len(tgt_test):
        print(f"❌ {impl_name}: Result length mismatch!")
        print(f"  Reference: {len(src_ref)} source points, {len(tgt_ref)} target points")
        print(f"  Test: {len(src_test)} source points, {len(tgt_test)} target points")
        return False

    # Check if the results contain the same indices
    src_ref_sorted = torch.sort(src_ref)[0]
    src_test_sorted = torch.sort(src_test)[0]
    tgt_ref_sorted = torch.sort(tgt_ref)[0]
    tgt_test_sorted = torch.sort(tgt_test)[0]

    src_match = torch.all(src_ref_sorted == src_test_sorted)
    tgt_match = torch.all(tgt_ref_sorted == tgt_test_sorted)

    if not src_match or not tgt_match:
        print(f"❌ {impl_name}: Result indices don't match!")
        if not src_match:
            print(f"  Source indices mismatch: {src_ref_sorted} vs {src_test_sorted}")
        if not tgt_match:
            print(f"  Target indices mismatch: {tgt_ref_sorted} vs {tgt_test_sorted}")
        return False

    print(f"✅ {impl_name}: Results match the reference implementation")
    return True


def get_gpu_info() -> str:
    """
    Get information about the GPU.

    Returns:
        String with GPU information
    """
    device = torch.cuda.get_device_name(0)
    return f"GPU: {device}"


def benchmark_implementations(
    sizes: List[int], radius: float = 0.1, num_runs: int = 3,
) -> Dict[str, List[float]]:
    """
    Benchmark different implementations with varying point cloud sizes.

    Args:
        sizes: List of point cloud sizes to benchmark
        radius: Distance radius for considering points as overlapping
        num_runs: Number of runs for each size

    Returns:
        Dictionary with timing results for each implementation
    """
    # Get GPU information
    gpu_info = get_gpu_info()
    print(f"Hardware: {gpu_info}")

    # Define implementations using partial
    implementations = {
        'tensor_cpu': partial(tensor_intersection, device='cpu'),
        'tensor_gpu': partial(tensor_intersection, device='cuda'),
        'tensor_gpu_recursive': partial(tensor_intersection_recursive, device='cuda')
    }

    results = {name: [] for name in implementations}
    results['kdtree'] = []  # Add KD-tree separately

    for size in sizes:
        print(f"Benchmarking with size {size}...")

        # Generate point clouds
        src_points, tgt_points = generate_random_point_clouds(size, size, radius=radius, device='cpu')

        # Get reference result from KD-tree implementation
        reference_result = kdtree_intersection(src_points, tgt_points, radius)

        # Time the KD-tree implementation
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            kdtree_intersection(src_points, tgt_points, radius)
            end_time = time.time()
            times.append(end_time - start_time)

        # Calculate average time for KD-tree
        avg_time = sum(times) / len(times)
        results['kdtree'].append(avg_time)
        print(f"  kdtree: {avg_time:.4f} seconds")

        # Run benchmarks for tensor implementations
        for impl_name, impl_func in implementations.items():
            try:
                # Run the implementation and verify results
                test_result = impl_func(src_points, tgt_points, radius)

                # For GPU implementation, convert results to CPU for verification
                if impl_name in ['tensor_gpu', 'tensor_gpu_recursive']:
                    test_result = (test_result[0].cpu(), test_result[1].cpu())

                # Verify results
                verify_results(reference_result, test_result, impl_name)

                # Time the implementation
                times = []
                for _ in range(num_runs):
                    if impl_name in ['tensor_gpu', 'tensor_gpu_recursive']:
                        # Synchronize GPU before timing
                        torch.cuda.synchronize()
                        start_time = time.time()
                        impl_func(src_points, tgt_points, radius)
                        # Synchronize GPU after computation
                        torch.cuda.synchronize()
                        end_time = time.time()
                    else:
                        start_time = time.time()
                        impl_func(src_points, tgt_points, radius)
                        end_time = time.time()

                    times.append(end_time - start_time)

                # Calculate average time
                avg_time = sum(times) / len(times)
                results[impl_name].append(avg_time)
                print(f"  {impl_name}: {avg_time:.4f} seconds")

            except torch.cuda.OutOfMemoryError:
                # Handle OOM error
                print(f"  {impl_name}: CUDA Out of Memory Error with size {size}")
                # Add None to indicate OOM for this size
                results[impl_name].append(None)
                # Clear CUDA cache to free memory
                torch.cuda.empty_cache()

    return results


def plot_results(sizes: List[int], results: Dict[str, List[float]], save_path: str = None):
    """
    Plot benchmark results.

    Args:
        sizes: List of point cloud sizes
        results: Dictionary with timing results for each implementation
        save_path: Path to save the plot (if None, display the plot)
    """
    plt.figure(figsize=(10, 6))

    for impl_name, times in results.items():
        # Filter out None values (OOM cases)
        valid_sizes = [size for i, size in enumerate(sizes) if times[i] is not None]
        valid_times = [time for time in times if time is not None]

        if valid_sizes:  # Only plot if we have valid data points
            plt.plot(valid_sizes, valid_times, marker='o', label=impl_name)

            # Add OOM annotations if any
            oom_sizes = [size for i, size in enumerate(sizes) if times[i] is None]
            for oom_size in oom_sizes:
                plt.annotate('OOM',
                            xy=(oom_size, plt.ylim()[1]),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center',
                            va='bottom',
                            arrowprops=dict(arrowstyle='->', color='red'))

    plt.xlabel('Point Cloud Size')
    plt.ylabel('Time (seconds)')
    plt.title('Point Cloud Intersection Benchmark')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    """Main function to run benchmarks."""
    # Define point cloud sizes to benchmark (10^3 to 10^6)
    sizes = [int(1e3), int(1e4), int(1e5), int(1e6)]

    # Run benchmarks
    results = benchmark_implementations(sizes)

    # Plot results
    plot_results(sizes, results, save_path='benchmarks/point_cloud_intersection_benchmark.png')

    # Print summary
    print("\nSummary:")
    for impl_name, times in results.items():
        print(f"{impl_name}:")
        for i, (size, time) in enumerate(zip(sizes, times)):
            if time is None:
                print(f"  Size {size}: OOM (Out of Memory)")
            else:
                print(f"  Size {size}: {time:.4f} seconds")


if __name__ == "__main__":
    main()
