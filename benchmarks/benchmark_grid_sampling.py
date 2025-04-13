import time
import torch
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from utils.point_cloud_ops.sampling.grid_sampling_3d import GridSampling3D as GridSampling3Dv1
from utils.point_cloud_ops.sampling.grid_sampling_3d_v2 import GridSampling3D as GridSampling3Dv2


def create_test_point_cloud(num_points: int, device: torch.device = None) -> Dict[str, torch.Tensor]:
    """Create a test point cloud with random positions and features.

    Args:
        num_points: Number of points in the point cloud
        device: Device to place tensors on

    Returns:
        Dictionary containing point cloud data
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create random positions
    pos = torch.rand(num_points, 3, device=device)

    # Create random features
    features = torch.rand(num_points, 10, device=device)

    # Create random categorical data
    change_map = torch.randint(0, 5, (num_points,), device=device)

    return {
        'pos': pos,
        'features': features,
        'change_map': change_map
    }


def compare_results(original_result: Dict[str, torch.Tensor],
                   optimized_result: Dict[str, torch.Tensor],
                   rtol: float = 1e-5,
                   atol: float = 1e-5) -> None:
    """Compare results from original and optimized implementations.

    Args:
        original_result: Result from original implementation
        optimized_result: Result from optimized implementation
        rtol: Relative tolerance for floating point comparison
        atol: Absolute tolerance for floating point comparison

    Raises:
        AssertionError: If results are not equivalent
    """
    # Check that both results have the same keys
    assert original_result.keys() == optimized_result.keys(), "Results have different keys"

    # Compare each key
    for key in original_result:
        assert key in optimized_result, f"Key '{key}' missing in optimized result"

        original_val = original_result[key]
        optimized_val = optimized_result[key]

        # Handle None values
        if original_val is None and optimized_val is None:
            continue

        assert original_val is not None and optimized_val is not None, f"Key '{key}' is None in one implementation but not the other"

        # Compare tensors
        if isinstance(original_val, torch.Tensor) and isinstance(optimized_val, torch.Tensor):
            # Check shapes
            assert original_val.shape == optimized_val.shape, f"Key '{key}' has different shapes: {original_val.shape} vs {optimized_val.shape}"

            # Check data types
            assert original_val.dtype == optimized_val.dtype, f"Key '{key}' has different dtypes: {original_val.dtype} vs {optimized_val.dtype}"

            # Compare values
            assert torch.allclose(original_val, optimized_val, rtol=rtol, atol=atol), f"Key '{key}' values do not match within tolerance"
        else:
            # For non-tensor values
            assert original_val == optimized_val, f"Key '{key}' values do not match"


def benchmark_grid_sampling(
    point_cloud_sizes: List[int] = [1000, 10000],
    voxel_sizes: List[float] = [0.1, 0.05, 0.01],
    modes: List[str] = ["mean"],
    device: torch.device = None,
    num_runs: int = 5
) -> Dict:
    """Benchmark original and optimized grid sampling implementations.

    Args:
        point_cloud_sizes: List of point cloud sizes to test
        voxel_sizes: List of voxel sizes to test
        modes: List of modes to test
        device: Device to run benchmarks on
        num_runs: Number of runs for each configuration

    Returns:
        Dictionary containing benchmark results
    """
    print("Starting benchmark...")  # Debug print
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")  # Debug print

    results = {
        'original': {},
        'optimized': {},
        'speedup': {}
    }

    for size in point_cloud_sizes:
        print(f"\nTesting point cloud size: {size}")  # Debug print
        for voxel_size in voxel_sizes:
            print(f"  Testing voxel size: {voxel_size}")  # Debug print
            for mode in modes:
                print(f"    Testing mode: {mode}")  # Debug print
                config_key = f"size_{size}_voxel_{voxel_size}_mode_{mode}"

                # Create test point cloud
                point_cloud = create_test_point_cloud(size, device)

                # Initialize samplers
                original_sampler = GridSampling3Dv1(size=voxel_size, mode=mode, device=device)
                optimized_sampler = GridSampling3Dv2(size=voxel_size, mode=mode, device=device)

                # Run benchmarks
                original_times = []
                optimized_times = []

                for _ in range(num_runs):
                    # Original implementation
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                    original_result = original_sampler(point_cloud)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    original_times.append(time.time() - start_time)

                    # Optimized implementation
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                    optimized_result = optimized_sampler(point_cloud)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    optimized_times.append(time.time() - start_time)

                # Verify equivalence
                compare_results(original_result, optimized_result)

                # Calculate average times
                original_avg_time = np.mean(original_times)
                optimized_avg_time = np.mean(optimized_times)

                # Store results
                results['original'][config_key] = original_avg_time
                results['optimized'][config_key] = optimized_avg_time
                results['speedup'][config_key] = original_avg_time / optimized_avg_time

                print(f"Config: {config_key}")
                print(f"  Original: {original_avg_time:.6f} s")
                print(f"  Optimized: {optimized_avg_time:.6f} s")
                print(f"  Speedup: {results['speedup'][config_key]:.2f}x")
                print()

    return results


def plot_results(results: Dict) -> None:
    """Plot benchmark results.

    Args:
        results: Dictionary containing benchmark results
    """
    # Extract configurations
    configs = list(results['original'].keys())

    # Extract point cloud sizes
    sizes = sorted(list(set([int(config.split('_')[1]) for config in configs])))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot execution time
    ax = axes[0]
    x = np.arange(len(sizes))
    width = 0.35

    # Only plot for 'mean' mode
    mode = 'mean'
    original_times = [results['original'][f"size_{size}_voxel_0.1_mode_{mode}"] for size in sizes]
    optimized_times = [results['optimized'][f"size_{size}_voxel_0.1_mode_{mode}"] for size in sizes]

    ax.bar(x, original_times, width, label=f'Original ({mode})')
    ax.bar(x + width, optimized_times, width, label=f'Optimized ({mode})')

    ax.set_xlabel('Point Cloud Size')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Execution Time Comparison')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(sizes)
    ax.legend()

    # Plot speedup
    ax = axes[1]
    speedups = [results['speedup'][f"size_{size}_voxel_0.1_mode_{mode}"] for size in sizes]
    ax.plot(x + width / 2, speedups, 'o-', label=f'Speedup ({mode})')

    ax.set_xlabel('Point Cloud Size')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Speedup Comparison')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(sizes)
    ax.legend()

    plt.tight_layout()
    plt.savefig('grid_sampling_benchmark.png')
    plt.show()


if __name__ == "__main__":
    try:
        # Run benchmarks with 10^3 and 10^4 points
        results = benchmark_grid_sampling(
            point_cloud_sizes=[1000, 10000],  # 10^3 and 10^4
            voxel_sizes=[0.1],
            modes=["mean"],
            num_runs=2
        )

        print("\nPlotting results...")
        # Plot results
        plot_results(results)

        # Print summary
        print("\nSummary:")
        print(f"Average speedup: {np.mean(list(results['speedup'].values())):.2f}x")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise
