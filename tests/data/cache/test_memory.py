import pytest
import torch
import psutil
from data.cache import DatasetCache


def test_cache_memory_management():
    """Test memory management and eviction."""
    tensor_dim = 1024
    num_channels = 3  # RGB images
    num_items_to_keep = 3  # We want the cache to hold exactly 3 items
    
    # Create a sample datapoint to calculate memory
    sample_tensor = torch.randn(num_channels, tensor_dim, tensor_dim, dtype=torch.float32)
    sample_datapoint = {
        'inputs': {'image': sample_tensor},
        'labels': {'class': torch.tensor([0])},
        'meta_info': {'filename': 'test_0.jpg'}
    }
    
    # Calculate memory needed for exactly num_items_to_keep items
    item_memory = DatasetCache._calculate_item_memory(sample_datapoint)
    total_memory_needed = item_memory * num_items_to_keep
    
    # Set percentage to allow exactly num_items_to_keep items
    max_percent = (total_memory_needed / psutil.virtual_memory().total) * 100
    
    # Initialize cache with calculated memory limit
    cache = DatasetCache(max_memory_percent=max_percent)
    
    # Create and store initial items
    tensors = []  # Keep references to prevent garbage collection
    for i in range(num_items_to_keep):
        tensor = torch.randn(num_channels, tensor_dim, tensor_dim, dtype=torch.float32)
        tensors.append(tensor)
        
        datapoint = {
            'inputs': {'image': tensor},
            'labels': {'class': torch.tensor([i])},
            'meta_info': {'filename': f'test_{i}.jpg'}
        }
        cache.put(i, datapoint)
    
    # Verify initial state
    assert len(cache.cache) == num_items_to_keep, f"Cache should have exactly {num_items_to_keep} items"
    
    # Add one more item to trigger eviction
    extra_tensor = torch.randn(num_channels, tensor_dim, tensor_dim, dtype=torch.float32)
    tensors.append(extra_tensor)
    
    extra_datapoint = {
        'inputs': {'image': extra_tensor},
        'labels': {'class': torch.tensor([num_items_to_keep])},
        'meta_info': {'filename': f'test_{num_items_to_keep}.jpg'}
    }
    cache.put(num_items_to_keep, extra_datapoint)
    
    # Verify eviction
    assert len(cache.cache) == num_items_to_keep, f"Cache should maintain {num_items_to_keep} items"
    assert 0 not in cache.cache, "First item should have been evicted"
    assert num_items_to_keep in cache.cache, "New item should be present"
    
    # Cleanup
    del tensors
    import gc
    gc.collect()
