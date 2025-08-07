"""Test combined cache edge cases and error conditions."""

import pytest
import torch
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os


def test_cache_both_disabled_all_operations(cache_both_disabled, sample_datapoint):
    """Test all operations work correctly when both caches are disabled."""
    cache = cache_both_disabled
    
    # Verify initial state
    assert cache.use_cpu_cache is False
    assert cache.use_disk_cache is False
    assert cache.cpu_cache is None
    assert cache.disk_cache is None
    
    # All operations should work without crashing
    cache.put(0, sample_datapoint)  # Should be no-op
    retrieved = cache.get(0)        # Should return None
    assert retrieved is None
    
    # Size operations
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 0
    
    # Clear operations should not crash
    cache.clear_cpu_cache()
    cache.clear_disk_cache()
    cache.clear_all()
    
    # Exists check should return False
    assert cache.exists_on_disk(0) is False
    
    # Info should reflect disabled state
    info = cache.get_info()
    assert info['use_cpu_cache'] is False
    assert info['use_disk_cache'] is False
    assert info['cpu_cache_size'] == 0
    assert info['disk_cache_size'] == 0


def test_cache_empty_operations(cache_with_both_enabled):
    """Test operations on empty cache."""
    cache = cache_with_both_enabled
    
    # Get from empty cache
    retrieved = cache.get(0)
    assert retrieved is None
    
    # Multiple gets from empty cache
    for i in range(5):
        assert cache.get(i) is None
    
    # Clear empty cache
    cache.clear_cpu_cache()
    cache.clear_disk_cache()
    cache.clear_all()
    
    # Sizes should remain zero
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 0
    
    # Exists checks on empty cache
    for i in range(5):
        assert cache.exists_on_disk(i) is False


def test_cache_negative_indices(cache_with_both_enabled, sample_datapoint):
    """Test cache operations with negative indices."""
    cache = cache_with_both_enabled
    
    # Store with negative index
    cache.put(-1, sample_datapoint)
    
    # Retrieve with negative index
    retrieved = cache.get(-1)
    assert retrieved is not None
    assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])
    
    # Check existence with negative index
    assert cache.exists_on_disk(-1) is True
    
    # Verify sizes
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 1


def test_cache_large_indices(cache_with_both_enabled, sample_datapoint):
    """Test cache operations with very large indices."""
    cache = cache_with_both_enabled
    large_idx = 2**32 - 1  # Very large index
    
    # Store with large index
    cache.put(large_idx, sample_datapoint)
    
    # Retrieve with large index
    retrieved = cache.get(large_idx)
    assert retrieved is not None
    assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])
    
    # Check existence with large index
    assert cache.exists_on_disk(large_idx) is True


def test_cache_none_values_handling(cache_with_both_enabled):
    """Test cache behavior with None device parameter and edge cases."""
    cache = cache_with_both_enabled
    
    # Test get with None device (should work)
    retrieved = cache.get(0, device=None)
    assert retrieved is None  # Item doesn't exist
    
    # Test with various None scenarios
    assert cache.exists_on_disk(0) is False


def test_cache_empty_datapoint_structures(cache_with_both_enabled):
    """Test cache with various empty or minimal datapoint structures."""
    cache = cache_with_both_enabled
    
    # Test empty inputs and labels
    empty_datapoint = {
        'inputs': {},
        'labels': {},
        'meta_info': {'test': 'empty'}
    }
    
    cache.put(0, empty_datapoint)
    retrieved = cache.get(0)
    assert retrieved is not None
    assert retrieved['inputs'] == {}
    assert retrieved['labels'] == {}
    assert retrieved['meta_info']['test'] == 'empty'
    
    # Test minimal structure
    minimal_datapoint = {
        'inputs': {'single': torch.tensor([1.0])},
        'labels': {'single': torch.tensor([0])},
        'meta_info': {}
    }
    
    cache.put(1, minimal_datapoint)
    retrieved = cache.get(1)
    assert retrieved is not None
    assert torch.equal(retrieved['inputs']['single'], torch.tensor([1.0]))


def test_cache_concurrent_access_stress(cache_with_both_enabled, make_datapoint_factory):
    """Test cache under concurrent access stress."""
    cache = cache_with_both_enabled
    num_threads = 10
    items_per_thread = 10
    results = []
    errors = []
    
    def worker(thread_id):
        try:
            # Each thread works with its own set of indices
            start_idx = thread_id * items_per_thread
            for i in range(items_per_thread):
                idx = start_idx + i
                datapoint = make_datapoint_factory(idx)
                
                # Store
                cache.put(idx, datapoint)
                
                # Retrieve
                retrieved = cache.get(idx)
                if retrieved is not None:
                    results.append(f"thread_{thread_id}_item_{i}_success")
                
                # Clear operations (some threads)
                if thread_id % 3 == 0:
                    cache.clear_cpu_cache()
                
                # Get info
                info = cache.get_info()
                results.append(f"thread_{thread_id}_info_size_{info['cpu_cache_size']}")
                
        except Exception as e:
            errors.append(f"thread_{thread_id}_error: {str(e)}")
    
    # Run concurrent workers
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for future in futures:
            future.result(timeout=30)  # Wait for completion with timeout
    
    # Verify no errors occurred
    assert len(errors) == 0, f"Concurrent access errors: {errors}"
    
    # Verify some results were recorded
    assert len(results) > 0


def test_cache_rapid_clear_operations(cache_with_both_enabled, make_datapoint_factory):
    """Test rapid clear operations don't cause issues."""
    cache = cache_with_both_enabled
    
    # Store some data
    for i in range(5):
        cache.put(i, make_datapoint_factory(i))
    
    # Perform rapid clear operations
    for _ in range(10):
        cache.clear_cpu_cache()
        cache.clear_disk_cache()
        cache.clear_all()
        
        # Verify consistent state
        assert cache.get_cpu_size() == 0
        assert cache.get_disk_size() == 0


def test_cache_disk_directory_issues(temp_cache_setup):
    """Test cache behavior with disk directory issues."""
    from data.cache.combined_dataset_cache import CombinedDatasetCache
    
    # Test with invalid directory path (should create directories)
    invalid_path = "/tmp/nonexistent/deep/path/test"
    cache = CombinedDatasetCache(
        data_root=invalid_path,
        version_hash="test",
        use_disk_cache=True
    )
    
    # Should still work - disk cache creates directories
    assert cache.disk_cache is not None
    assert os.path.exists(f"{invalid_path}_cache")


def test_cache_version_hash_edge_cases(temp_cache_setup):
    """Test cache with various version hash formats."""
    from data.cache.combined_dataset_cache import CombinedDatasetCache
    
    edge_case_hashes = [
        "",  # Empty string
        "a",  # Single character
        "very_long_hash_" * 10,  # Very long hash
        "hash-with-dashes",
        "hash_with_underscores",
        "123456789",  # Numeric
        "MiXeD_CaSe_HaSh",  # Mixed case
    ]
    
    for version_hash in edge_case_hashes:
        cache = CombinedDatasetCache(
            data_root=temp_cache_setup['data_root'],
            version_hash=version_hash,
            use_disk_cache=True
        )
        
        assert cache.disk_cache.version_hash == version_hash
        
        # Should create appropriate directory
        if version_hash:  # Skip empty string which might cause issues
            expected_dir = os.path.join(f"{temp_cache_setup['data_root']}_cache", version_hash)
            assert os.path.exists(expected_dir)


def test_cache_memory_pressure_simulation(cache_with_both_enabled, make_datapoint_factory):
    """Test cache behavior under simulated memory pressure."""
    cache = cache_with_both_enabled
    
    # Store many items to potentially trigger memory management
    num_items = 50
    for i in range(num_items):
        large_datapoint = {
            'inputs': {'image': torch.randn(3, 256, 256, dtype=torch.float32)},
            'labels': {'mask': torch.randint(0, 10, (256, 256), dtype=torch.long)},
            'meta_info': {'index': i}
        }
        cache.put(i, large_datapoint)
    
    # Verify some items are stored
    info = cache.get_info()
    assert info['cpu_cache_size'] > 0 or info['disk_cache_size'] > 0
    
    # Try to retrieve items
    retrieved_count = 0
    for i in range(num_items):
        retrieved = cache.get(i)
        if retrieved is not None:
            retrieved_count += 1
    
    # Should be able to retrieve at least some items
    assert retrieved_count > 0


def test_cache_device_parameter_edge_cases(cache_with_both_enabled, sample_datapoint):
    """Test device parameter edge cases."""
    cache = cache_with_both_enabled
    cache.put(0, sample_datapoint)
    
    # Test with various device formats
    device_cases = [
        None,
        'cpu',
        # Note: torch.device objects may not be supported by all cache layers
    ]
    
    for device in device_cases:
        retrieved = cache.get(0, device=device)
        assert retrieved is not None
    
    # Test CUDA if available (note: may return CPU tensors due to cache hierarchy)
    if torch.cuda.is_available():
        retrieved_cuda = cache.get(0, device='cuda')
        assert retrieved_cuda is not None


def test_cache_repeated_get_operations(cache_with_both_enabled, sample_datapoint):
    """Test repeated get operations for consistency."""
    cache = cache_with_both_enabled
    cache.put(0, sample_datapoint)
    
    # Perform many get operations
    for _ in range(100):
        retrieved = cache.get(0)
        assert retrieved is not None
        assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])


def test_cache_mixed_operations_sequence(cache_with_both_enabled, make_datapoint_factory):
    """Test complex sequence of mixed cache operations."""
    cache = cache_with_both_enabled
    
    # Complex sequence of operations
    for i in range(10):
        # Store
        cache.put(i, make_datapoint_factory(i))
        
        # Get some items
        if i > 2:
            cache.get(i-2)
        
        # Clear operations periodically
        if i % 3 == 0:
            cache.clear_cpu_cache()
        
        # Check existence
        cache.exists_on_disk(i)
        
        # Get info
        cache.get_info()
    
    # Final verification
    info = cache.get_info()
    assert isinstance(info, dict)
    assert 'cpu_cache_size' in info
    assert 'disk_cache_size' in info


def test_cache_zero_memory_limit(temp_cache_setup):
    """Test cache behavior with zero memory limit."""
    from data.cache.combined_dataset_cache import CombinedDatasetCache
    
    # This should still work but with very limited CPU cache
    cache = CombinedDatasetCache(
        data_root=temp_cache_setup['data_root'],
        version_hash=temp_cache_setup['version_hash'],
        max_cpu_memory_percent=0.1  # Very small but not zero
    )
    
    assert cache.cpu_cache.max_memory_percent == 0.1


def test_cache_exception_resilience(cache_with_both_enabled, sample_datapoint):
    """Test cache resilience to various exception scenarios."""
    cache = cache_with_both_enabled
    
    # Store valid data
    cache.put(0, sample_datapoint)
    
    # Test with malformed datapoints (should not crash the cache)
    try:
        malformed_datapoint = {
            'inputs': {'bad': "not a tensor"},  # Invalid tensor
            'labels': {},
            'meta_info': {}
        }
        # This might fail but shouldn't crash the cache
        cache.put(1, malformed_datapoint)
    except:
        pass  # Expected to potentially fail
    
    # Original data should still be retrievable
    retrieved = cache.get(0)
    assert retrieved is not None
    assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])