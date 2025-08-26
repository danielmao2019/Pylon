"""Single method benchmarking utility for KNN."""

import time
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from utils.point_cloud_ops.knn.knn import knn


def benchmark_knn_method(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    method: str,
    k: int = 10,
    warmup_runs: int = 1,
    test_runs: int = 3
) -> float:
    """Benchmark a single KNN method.
    
    Args:
        query_points: Query point cloud
        reference_points: Reference point cloud
        method: KNN method name
        k: Number of nearest neighbors
        warmup_runs: Number of warmup runs
        test_runs: Number of test runs for timing
        
    Returns:
        Average runtime in seconds
    """
    # Warmup runs
    for _ in range(warmup_runs):
        try:
            _ = knn(
                query_points=query_points,
                reference_points=reference_points,
                k=k,
                method=method,
                return_distances=True
            )
        except Exception as e:
            print(f"Warning: {method} failed during warmup: {e}")
            return float('inf')
    
    # Test runs
    times = []
    for _ in range(test_runs):
        try:
            start_time = time.perf_counter()
            _ = knn(
                query_points=query_points,
                reference_points=reference_points,
                k=k,
                method=method,
                return_distances=True
            )
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        except Exception as e:
            print(f"Warning: {method} failed during test: {e}")
            return float('inf')
    
    return np.mean(times)
