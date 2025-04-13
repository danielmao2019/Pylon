#!/usr/bin/env python
"""
Benchmark script for comparing different point cloud intersection implementations.
"""
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the original implementation
from utils.point_cloud_ops.set_ops.intersection import pc_intersection as original_pc_intersection


def tensor_ops_pc_intersection(
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


def kdtree_pc_intersection(
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


def original_loop_pc_intersection(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Original implementation with for loops and tensor operations.
    
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
    
    # Count source points that are close to any target point
    src_overlapping_indices = []
    for i, src_point in enumerate(src_points):
        # Find points in the target point cloud that are close to this source point
        distances = torch.norm(tgt_points - src_point, dim=1)
        close_points = torch.where(distances < radius)[0]
        if len(close_points) > 0:
            src_overlapping_indices.append(i)
    
    # Count target points that are close to any source point
    tgt_overlapping_indices = []
    for i, tgt_point in enumerate(tgt_points):
        # Find points in the source point cloud that are close to this target point
        distances = torch.norm(src_points - tgt_point, dim=1)
        close_points = torch.where(distances < radius)[0]
        if len(close_points) > 0:
            tgt_overlapping_indices.append(i)
    
    # Convert lists to tensors
    src_overlapping_indices = torch.tensor(src_overlapping_indices, device=src_points.device)
    tgt_overlapping_indices = torch.tensor(tgt_overlapping_indices, device=tgt_points.device)
    
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


def benchmark_implementations(sizes: List[int], radius: float = 0.1, 
                             num_runs: int = 5, device: str = 'cpu') -> Dict[str, List[float]]:
    """
    Benchmark different implementations with varying point cloud sizes.
    
    Args:
        sizes: List of point cloud sizes to benchmark
        radius: Distance radius for considering points as overlapping
        num_runs: Number of runs for each size
        device: Device to run benchmarks on
        
    Returns:
        Dictionary with timing results for each implementation
    """
    implementations = {
        'Original (Loops)': original_loop_pc_intersection,
        'KD-Tree': kdtree_pc_intersection,
        'Tensor Ops': tensor_ops_pc_intersection
    }
    
    results = {name: [] for name in implementations}
    
    for size in sizes:
        print(f"Benchmarking with size {size}...")
        
        # Generate point clouds
        src_points, tgt_points = generate_random_point_clouds(size, size, radius=radius, device=device)
        
        # Run benchmarks
        for impl_name, impl_func in implementations.items():
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                impl_func(src_points, tgt_points, radius)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate average time
            avg_time = sum(times) / len(times)
            results[impl_name].append(avg_time)
            print(f"  {impl_name}: {avg_time:.4f} seconds")
    
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
        plt.plot(sizes, times, marker='o', label=impl_name)
    
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
    # Define point cloud sizes to benchmark
    sizes = [100, 500, 1000, 5000, 10000]
    
    # Run benchmarks
    results = benchmark_implementations(sizes)
    
    # Plot results
    plot_results(sizes, results, save_path='benchmarks/point_cloud_intersection_benchmark.png')
    
    # Print summary
    print("\nSummary:")
    for impl_name, times in results.items():
        print(f"{impl_name}:")
        for size, time in zip(sizes, times):
            print(f"  Size {size}: {time:.4f} seconds")


if __name__ == "__main__":
    main()
