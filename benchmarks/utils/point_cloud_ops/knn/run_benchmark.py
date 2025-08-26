"""Main benchmark execution logic for KNN methods."""

import torch
import numpy as np
from typing import Dict
from generate_point_clouds import generate_point_clouds
from benchmark_method import benchmark_knn_method


def run_benchmark() -> Dict:
    """Run the complete benchmark.
    
    Returns:
        Dictionary with benchmark results
    """
    # Configuration
    sizes = [1_000, 10_000, 100_000, 1_000_000]
    size_labels = ["1K", "10K", "100K", "1M"]
    methods = ["faiss", "pytorch3d", "torch", "scipy"]
    shapes = ["uniform_cube", "gaussian_cluster", "sphere_surface", "line_with_noise", "multiple_clusters"]
    k = 10  # Number of nearest neighbors to find
    
    # Check CUDA availability
    assert torch.cuda.is_available(), "CUDA must be available for GPU-based KNN methods (faiss, pytorch3d)"
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Results storage
    results = {
        "sizes": sizes,
        "size_labels": size_labels,
        "methods": methods,
        "shapes": shapes,
        "times": {}  # method -> list of times for each size
    }
    
    for method in methods:
        results["times"][method] = []
    
    # Run benchmarks
    total_tests = len(sizes) * len(methods) * len(shapes) * 3  # 3 repetitions
    test_count = 0
    
    for size_idx, (size, size_label) in enumerate(zip(sizes, size_labels)):
        print(f"\n{'='*60}")
        print(f"Benchmarking size: {size_label} ({size:,} points)")
        print(f"{'='*60}")
        
        # Generate point clouds for this size
        print(f"Generating {len(shapes)} different point cloud shapes...")
        point_clouds = generate_point_clouds(size, shapes, device)
        
        # Benchmark each method
        method_times = {method: [] for method in methods}
        
        for method in methods:
            print(f"\nTesting method: {method}")
            
            for shape_idx, (shape_name, (query, reference)) in enumerate(zip(shapes, point_clouds)):
                test_count += 3  # 3 repetitions
                print(f"  Shape {shape_idx+1}/{len(shapes)}: {shape_name} - ", end="", flush=True)
                
                # Move to CPU for scipy
                if method == "scipy":
                    query_cpu = query.cpu()
                    reference_cpu = reference.cpu()
                    avg_time = benchmark_knn_method(query_cpu, reference_cpu, method, k=k)
                else:
                    avg_time = benchmark_knn_method(query, reference, method, k=k)
                
                if avg_time == float('inf'):
                    print("FAILED")
                else:
                    print(f"{avg_time:.4f}s")
                    method_times[method].append(avg_time)
                
                print(f"  Progress: {test_count}/{total_tests} tests completed", end="\r")
        
        # Calculate average time for each method at this size
        for method in methods:
            if method_times[method]:
                avg_time = np.mean(method_times[method])
                results["times"][method].append(avg_time)
            else:
                results["times"][method].append(float('inf'))
            
            print(f"{method}: {results['times'][method][-1]:.4f}s average")
    
    print(f"\n{'='*60}")
    print("Benchmark completed!")
    
    return results