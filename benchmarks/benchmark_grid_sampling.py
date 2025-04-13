import time
import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from utils.point_cloud_ops.sampling.grid_sampling_3d import GridSampling3D as OriginalGridSampling3D
from utils.point_cloud_ops.sampling.grid_sampling_3d_v2 import GridSampling3D as OptimizedGridSampling3D


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
                   atol: float = 1e-5) -> Tuple[bool, Dict[str, bool]]:
    """Compare results from original and optimized implementations.
    
    Args:
        original_result: Result from original implementation
        optimized_result: Result from optimized implementation
        rtol: Relative tolerance for floating point comparison
        atol: Absolute tolerance for floating point comparison
        
    Returns:
        Tuple of (all_match, detailed_results)
    """
    # Check that both results have the same keys
    if original_result.keys() != optimized_result.keys():
        return False, {'keys_match': False}
    
    detailed_results = {'keys_match': True}
    all_match = True
    
    # Compare each key
    for key in original_result:
        if key not in optimized_result:
            detailed_results[key] = False
            all_match = False
            continue
        
        original_val = original_result[key]
        optimized_val = optimized_result[key]
        
        # Handle None values
        if original_val is None and optimized_val is None:
            detailed_results[key] = True
            continue
        
        if original_val is None or optimized_val is None:
            detailed_results[key] = False
            all_match = False
            continue
        
        # Compare tensors
        if isinstance(original_val, torch.Tensor) and isinstance(optimized_val, torch.Tensor):
            # Check shapes
            if original_val.shape != optimized_val.shape:
                detailed_results[key] = False
                all_match = False
                continue
            
            # Check data types
            if original_val.dtype != optimized_val.dtype:
                detailed_results[key] = False
                all_match = False
                continue
            
            # Compare values
            try:
                match = torch.allclose(original_val, optimized_val, rtol=rtol, atol=atol)
                detailed_results[key] = match
                if not match:
                    all_match = False
            except Exception as e:
                detailed_results[key] = False
                all_match = False
        else:
            # For non-tensor values
            detailed_results[key] = original_val == optimized_val
            if not detailed_results[key]:
                all_match = False
    
    return all_match, detailed_results


def benchmark_grid_sampling(
    point_cloud_sizes: List[int] = [1000, 10000, 100000],
    voxel_sizes: List[float] = [0.1, 0.05, 0.01],
    modes: List[str] = ["mean", "last"],
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
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {
        'original': {},
        'optimized': {},
        'equivalence': {},
        'speedup': {}
    }
    
    for size in point_cloud_sizes:
        for voxel_size in voxel_sizes:
            for mode in modes:
                config_key = f"size_{size}_voxel_{voxel_size}_mode_{mode}"
                
                # Create test point cloud
                point_cloud = create_test_point_cloud(size, device)
                
                # Initialize samplers
                original_sampler = OriginalGridSampling3D(size=voxel_size, mode=mode, device=device)
                optimized_sampler = OptimizedGridSampling3D(size=voxel_size, mode=mode, device=device)
                
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
                
                # Calculate average times
                original_avg_time = np.mean(original_times)
                optimized_avg_time = np.mean(optimized_times)
                
                # Store results
                results['original'][config_key] = original_avg_time
                results['optimized'][config_key] = optimized_avg_time
                results['speedup'][config_key] = original_avg_time / optimized_avg_time
                
                # Check equivalence
                all_match, detailed_results = compare_results(original_result, optimized_result)
                results['equivalence'][config_key] = {
                    'all_match': all_match,
                    'detailed_results': detailed_results
                }
                
                print(f"Config: {config_key}")
                print(f"  Original: {original_avg_time:.6f} s")
                print(f"  Optimized: {optimized_avg_time:.6f} s")
                print(f"  Speedup: {results['speedup'][config_key]:.2f}x")
                print(f"  Equivalent: {all_match}")
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
    
    for i, mode in enumerate(['mean', 'last']):
        original_times = [results['original'][f"size_{size}_voxel_0.1_mode_{mode}"] for size in sizes]
        optimized_times = [results['optimized'][f"size_{size}_voxel_0.1_mode_{mode}"] for size in sizes]
        
        ax.bar(x + i * width, original_times, width, label=f'Original ({mode})')
        ax.bar(x + i * width + width, optimized_times, width, label=f'Optimized ({mode})')
    
    ax.set_xlabel('Point Cloud Size')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Execution Time Comparison')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(sizes)
    ax.legend()
    
    # Plot speedup
    ax = axes[1]
    for i, mode in enumerate(['mean', 'last']):
        speedups = [results['speedup'][f"size_{size}_voxel_0.1_mode_{mode}"] for size in sizes]
        ax.plot(x + i * width + width / 2, speedups, 'o-', label=f'Speedup ({mode})')
    
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
    # Run benchmarks
    results = benchmark_grid_sampling(
        point_cloud_sizes=[1000, 10000, 100000],
        voxel_sizes=[0.1],
        modes=["mean", "last"],
        num_runs=3
    )
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print("\nSummary:")
    print(f"Average speedup: {np.mean(list(results['speedup'].values())):.2f}x")
    
    # Check equivalence
    all_equivalent = all(result['all_match'] for result in results['equivalence'].values())
    print(f"All results equivalent: {all_equivalent}")
    
    if not all_equivalent:
        print("\nNon-equivalent results:")
        for config, result in results['equivalence'].items():
            if not result['all_match']:
                print(f"  {config}:")
                for key, match in result['detailed_results'].items():
                    if not match:
                        print(f"    - {key} does not match")
