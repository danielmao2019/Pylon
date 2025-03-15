import pytest
import torch
import psutil
from data.cache import DatasetCache


def test_cache_memory_management():
    """Test memory management and eviction."""
    tensor_dim = 1024
    num_channels = 3  # RGB images
    memory_threshold = 0.01  # 0.01% increase target
    
    # Calculate required iterations (add 3 for safety)
    tensor_bytes = tensor_dim * tensor_dim * num_channels * 4  # 4 bytes per float32
    threshold_bytes = psutil.virtual_memory().total * (memory_threshold / 100)
    required_iterations = int(threshold_bytes / tensor_bytes) + 3
    
    # Initialize cache with memory limit
    initial_memory = psutil.Process().memory_percent()
    cache = DatasetCache(max_memory_percent=initial_memory + memory_threshold)
    
    print(f"\nTest parameters:")
    print(f"- Tensor size: {tensor_dim}x{tensor_dim}x{num_channels} (~{tensor_bytes/1024/1024:.1f}MB each)")
    print(f"- Memory threshold: {memory_threshold}% ({threshold_bytes/1024/1024:.1f}MB)")
    print(f"- Required iterations: {required_iterations}")
    
    # Create tensors that will consume memory
    tensors = []
    start_memory = psutil.Process().memory_percent()
    
    for iteration in range(required_iterations):
        # Create and store tensor
        tensor = torch.randn(num_channels, tensor_dim, tensor_dim, dtype=torch.float32)
        tensors.append(tensor)  # Keep reference to prevent garbage collection
        
        datapoint = {
            'inputs': {'image': tensor},
            'labels': {'class': torch.tensor([iteration])},
            'meta_info': {'filename': f'test_{iteration}.jpg'}
        }
        cache.put(iteration, datapoint)
        
        # Check memory state
        current_memory = psutil.Process().memory_percent()
        memory_increase = current_memory - start_memory
        
        print(f"Iteration {iteration}: Memory increase: {memory_increase:.3f}%")
        
        # Check if eviction occurred
        if 0 not in cache.cache:
            print(f"\nEviction occurred at iteration {iteration}")
            print(f"Memory increase at eviction: {memory_increase:.3f}%")
            break
    else:
        pytest.fail(
            f"No eviction after {required_iterations} iterations.\n"
            f"Memory increase: {memory_increase:.3f}%\n"
            f"Expected threshold: {memory_threshold}%\n"
            f"Cache size: {len(cache.cache)}"
        )
    
    # Verify final state
    assert len(cache.cache) > 0, "Cache should not be empty"
    assert 0 not in cache.cache, "First item should have been evicted"
    assert iteration <= required_iterations, "Took too many iterations"
    
    # Log final statistics
    print(f"\nFinal cache size: {len(cache.cache)} items")
    print(f"Final memory increase: {memory_increase:.3f}%")
    
    # Cleanup
    del tensors
    import gc
    gc.collect()
