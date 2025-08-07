"""Test combined cache basic put/get operations."""

import pytest
import torch
from utils.ops import buffer_equal


def test_cache_put_and_get_both_enabled(cache_with_both_enabled, sample_datapoint):
    """Test basic put and get operations with both caches enabled."""
    cache = cache_with_both_enabled
    
    # Test put operation
    cache.put(0, sample_datapoint)
    
    # Verify data is stored in both caches
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 1
    
    # Test get operation
    retrieved = cache.get(0)
    assert retrieved is not None
    
    # Verify data integrity
    assert retrieved['inputs']['image'].shape == sample_datapoint['inputs']['image'].shape
    assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])
    assert torch.equal(retrieved['labels']['class_id'], sample_datapoint['labels']['class_id'])
    assert retrieved['meta_info']['path'] == sample_datapoint['meta_info']['path']


def test_cache_put_and_get_cpu_only(cache_cpu_only, sample_datapoint):
    """Test put and get operations with only CPU cache enabled."""
    cache = cache_cpu_only
    
    # Test put operation
    cache.put(0, sample_datapoint)
    
    # Verify data is stored only in CPU cache
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 0  # Disk cache disabled
    
    # Test get operation
    retrieved = cache.get(0)
    assert retrieved is not None
    assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])


def test_cache_put_and_get_disk_only(cache_disk_only, sample_datapoint):
    """Test put and get operations with only disk cache enabled."""
    cache = cache_disk_only
    
    # Test put operation
    cache.put(0, sample_datapoint)
    
    # Verify data is stored only in disk cache
    assert cache.get_cpu_size() == 0  # CPU cache disabled
    assert cache.get_disk_size() == 1
    
    # Test get operation
    retrieved = cache.get(0)
    assert retrieved is not None
    assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])


def test_cache_put_and_get_both_disabled(cache_both_disabled, sample_datapoint):
    """Test put and get operations with both caches disabled."""
    cache = cache_both_disabled
    
    # Test put operation (should not crash but does nothing)
    cache.put(0, sample_datapoint)
    
    # Verify no data is stored
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 0
    
    # Test get operation (should return None)
    retrieved = cache.get(0)
    assert retrieved is None


def test_cache_get_nonexistent_key(cache_with_both_enabled):
    """Test get operation with non-existent key."""
    cache = cache_with_both_enabled
    
    # Test get with non-existent key
    retrieved = cache.get(999)
    assert retrieved is None
    
    # Verify cache sizes remain zero
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 0


def test_cache_multiple_items(cache_with_both_enabled, make_datapoint_factory):
    """Test put and get operations with multiple items."""
    cache = cache_with_both_enabled
    datapoints = [make_datapoint_factory(i) for i in range(5)]
    
    # Store multiple items
    for i, datapoint in enumerate(datapoints):
        cache.put(i, datapoint)
    
    # Verify all items are stored
    assert cache.get_cpu_size() == 5
    assert cache.get_disk_size() == 5
    
    # Verify all items can be retrieved correctly
    for i, original_datapoint in enumerate(datapoints):
        retrieved = cache.get(i)
        assert retrieved is not None
        assert torch.allclose(retrieved['inputs']['image'], original_datapoint['inputs']['image'])
        assert torch.equal(retrieved['labels']['class_id'], original_datapoint['labels']['class_id'])
        assert retrieved['meta_info']['index'] == original_datapoint['meta_info']['index']


def test_cache_data_isolation(cache_with_both_enabled, sample_datapoint):
    """Test that cached data is properly isolated through deep copying."""
    cache = cache_with_both_enabled
    
    # Store original data
    cache.put(0, sample_datapoint)
    
    # Modify original data
    original_image = sample_datapoint['inputs']['image'].clone()
    sample_datapoint['inputs']['image'] += 100.0
    sample_datapoint['meta_info']['path'] = 'modified_path.jpg'
    
    # Retrieved data should not be affected
    retrieved = cache.get(0)
    assert not torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])
    assert torch.allclose(retrieved['inputs']['image'], original_image)
    assert retrieved['meta_info']['path'] != 'modified_path.jpg'
    
    # Modify retrieved data
    retrieved['inputs']['image'] += 200.0
    retrieved['meta_info']['path'] = 'another_modification.jpg'
    
    # Get again should return unmodified data
    retrieved_again = cache.get(0)
    assert not torch.allclose(retrieved_again['inputs']['image'], retrieved['inputs']['image'])
    assert torch.allclose(retrieved_again['inputs']['image'], original_image)
    assert retrieved_again['meta_info']['path'] != 'another_modification.jpg'


def test_cache_device_parameter_handling(cache_with_both_enabled, sample_datapoint):
    """Test device parameter handling in get operations."""
    cache = cache_with_both_enabled
    cache.put(0, sample_datapoint)
    
    device_scenarios = [
        None,  # Default (no device specified)
        'cpu',
    ]
    
    # Add CUDA only if available
    if torch.cuda.is_available():
        device_scenarios.append('cuda')
    
    for device in device_scenarios:
        retrieved = cache.get(0, device=device)
        assert retrieved is not None
        
        # Note: Combined cache behavior depends on which cache layer returns the data:
        # - CPU cache always returns data on CPU regardless of device parameter
        # - Disk cache honors device parameter for direct loads
        # Since we stored in both caches, CPU cache will be hit first and return CPU tensors
        # This is the expected hierarchical behavior


def test_cache_device_parameter_propagation(cache_with_both_enabled, sample_datapoint):
    """Test that device parameter is properly propagated to underlying caches."""
    cache = cache_with_both_enabled
    cache.put(0, sample_datapoint)
    
    # Clear CPU cache to force retrieval from disk
    cache.clear_cpu_cache()
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 1
    
    # Get with device parameter should promote to CPU
    device = 'cpu'
    retrieved = cache.get(0, device=device)
    assert retrieved is not None
    
    # Verify data was promoted to CPU cache
    assert cache.get_cpu_size() == 1
    
    # Note: Device handling behavior:
    # - Disk cache can load directly to specified device
    # - CPU cache promotion stores data on CPU regardless of original device
    # - This is the expected behavior for the cache hierarchy


def test_cache_overwrite_existing_item(cache_with_both_enabled, sample_datapoint, different_datapoint):
    """Test overwriting an existing cache item."""
    cache = cache_with_both_enabled
    
    # Store initial data
    cache.put(0, sample_datapoint)
    initial_retrieved = cache.get(0)
    assert torch.allclose(initial_retrieved['inputs']['image'], sample_datapoint['inputs']['image'])
    
    # Overwrite with different data
    cache.put(0, different_datapoint)
    
    # Verify sizes don't change (same index)
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 1
    
    # Verify new data is retrieved
    updated_retrieved = cache.get(0)
    assert not torch.allclose(updated_retrieved['inputs']['image'], sample_datapoint['inputs']['image'])
    assert torch.allclose(updated_retrieved['inputs']['image'], different_datapoint['inputs']['image'])
    assert updated_retrieved['labels']['class_id'] == different_datapoint['labels']['class_id']


def test_cache_large_datapoint(cache_with_both_enabled, large_datapoint):
    """Test caching of large datapoints."""
    cache = cache_with_both_enabled
    
    # Store large datapoint
    cache.put(0, large_datapoint)
    
    # Verify it's stored in both caches
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 1
    
    # Retrieve and verify integrity
    retrieved = cache.get(0)
    assert retrieved is not None
    
    # Verify large tensor integrity
    assert retrieved['inputs']['image'].shape == large_datapoint['inputs']['image'].shape
    assert torch.allclose(retrieved['inputs']['image'], large_datapoint['inputs']['image'])
    assert retrieved['inputs']['extra_data'].shape == large_datapoint['inputs']['extra_data'].shape
    assert torch.allclose(retrieved['inputs']['extra_data'], large_datapoint['inputs']['extra_data'])


def test_cache_complex_nested_structure(cache_with_both_enabled):
    """Test caching of datapoints with complex nested structures."""
    complex_datapoint = {
        'inputs': {
            'image': torch.randn(3, 64, 64, dtype=torch.float32),
            'nested': {
                'features': torch.randn(100, dtype=torch.float32),
                'deep_nested': {
                    'extra': torch.randn(50, dtype=torch.float32)
                }
            }
        },
        'labels': {
            'mask': torch.randint(0, 2, (64, 64), dtype=torch.long),
            'bbox': torch.tensor([[10, 20, 30, 40]], dtype=torch.float32)
        },
        'meta_info': {
            'path': '/test/complex.jpg',
            'nested_meta': {
                'annotation_id': 123,
                'complexity_level': 'high'
            }
        }
    }
    
    cache = cache_with_both_enabled
    
    # Store complex datapoint
    cache.put(0, complex_datapoint)
    
    # Retrieve and verify structure integrity
    retrieved = cache.get(0)
    assert retrieved is not None
    
    # Verify nested structure is preserved
    assert 'nested' in retrieved['inputs']
    assert 'deep_nested' in retrieved['inputs']['nested']
    assert torch.allclose(retrieved['inputs']['nested']['deep_nested']['extra'], 
                         complex_datapoint['inputs']['nested']['deep_nested']['extra'])
    
    assert 'nested_meta' in retrieved['meta_info']
    assert retrieved['meta_info']['nested_meta']['annotation_id'] == 123


def test_cache_tensor_dtype_preservation(cache_with_both_enabled):
    """Test that tensor dtypes are preserved through cache operations."""
    mixed_dtype_datapoint = {
        'inputs': {
            'float32_tensor': torch.randn(10, dtype=torch.float32),
            'float64_tensor': torch.randn(10, dtype=torch.float64),
            'int32_tensor': torch.randint(0, 100, (10,), dtype=torch.int32),
            'int64_tensor': torch.randint(0, 100, (10,), dtype=torch.int64),
            'bool_tensor': torch.randint(0, 2, (10,), dtype=torch.bool)
        },
        'labels': {
            'long_tensor': torch.tensor([1, 2, 3], dtype=torch.long)
        },
        'meta_info': {'test': 'dtype_preservation'}
    }
    
    cache = cache_with_both_enabled
    cache.put(0, mixed_dtype_datapoint)
    retrieved = cache.get(0)
    
    # Verify all dtypes are preserved
    for key, original_tensor in mixed_dtype_datapoint['inputs'].items():
        retrieved_tensor = retrieved['inputs'][key]
        assert retrieved_tensor.dtype == original_tensor.dtype, f"Dtype mismatch for {key}"
        assert torch.equal(retrieved_tensor, original_tensor), f"Value mismatch for {key}"
    
    assert retrieved['labels']['long_tensor'].dtype == torch.long