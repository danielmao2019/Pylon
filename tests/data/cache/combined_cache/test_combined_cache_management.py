"""Test combined cache management operations (clear, size, info, exists)."""

import pytest
import torch
import os


def test_cache_clear_cpu_cache_only(cache_with_both_enabled, sample_datapoint):
    """Test clearing only the CPU cache while preserving disk cache."""
    cache = cache_with_both_enabled
    
    # Store data in both caches
    cache.put(0, sample_datapoint)
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 1
    
    # Clear only CPU cache
    cache.clear_cpu_cache()
    
    # Verify CPU cache is cleared but disk cache remains
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 1
    
    # Verify data can still be retrieved (from disk) and promotes to CPU
    retrieved = cache.get(0)
    assert retrieved is not None
    assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])
    
    # Verify promotion occurred
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 1


def test_cache_clear_disk_cache_only(cache_with_both_enabled, sample_datapoint):
    """Test clearing only the disk cache while preserving CPU cache."""
    cache = cache_with_both_enabled
    
    # Store data in both caches
    cache.put(0, sample_datapoint)
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 1
    
    # Clear only disk cache
    cache.clear_disk_cache()
    
    # Verify disk cache is cleared but CPU cache remains
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 0
    
    # Verify data can still be retrieved (from CPU)
    retrieved = cache.get(0)
    assert retrieved is not None
    assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])


def test_cache_clear_all_caches(cache_with_both_enabled, sample_datapoint):
    """Test clearing both CPU and disk caches."""
    cache = cache_with_both_enabled
    
    # Store data in both caches
    cache.put(0, sample_datapoint)
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 1
    
    # Clear all caches
    cache.clear_all()
    
    # Verify both caches are cleared
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 0
    
    # Verify data cannot be retrieved
    retrieved = cache.get(0)
    assert retrieved is None


def test_cache_clear_operations_with_disabled_caches(all_cache_configurations, cache_config_factory, sample_datapoint):
    """Test clear operations with different cache enable/disable configurations."""
    for use_cpu, use_disk, description in all_cache_configurations:
        cache = cache_config_factory(
            use_cpu_cache=use_cpu,
            use_disk_cache=use_disk
        )
        
        # Store data if possible
        cache.put(0, sample_datapoint)
        
        # Test all clear operations (should not crash)
        cache.clear_cpu_cache()
        cache.clear_disk_cache()
        cache.clear_all()
        
        # Verify sizes are zero after clearing
        assert cache.get_cpu_size() == 0, f"CPU size not zero after clear for {description}"
        assert cache.get_disk_size() == 0, f"Disk size not zero after clear for {description}"


def test_cache_size_reporting_accuracy(cache_with_both_enabled, make_datapoint_factory):
    """Test that cache size reporting is accurate."""
    cache = cache_with_both_enabled
    
    # Verify initial sizes
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 0
    
    # Add items one by one and verify size increments
    for i in range(5):
        datapoint = make_datapoint_factory(i)
        cache.put(i, datapoint)
        
        assert cache.get_cpu_size() == i + 1
        assert cache.get_disk_size() == i + 1
    
    # Remove from CPU cache by clearing
    cache.clear_cpu_cache()
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 5  # Disk remains
    
    # Promote one item back to CPU
    cache.get(0)  # Should promote from disk to CPU
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 5


def test_cache_size_with_different_configurations(all_cache_configurations, cache_config_factory, sample_datapoint):
    """Test size reporting with different cache configurations."""
    for use_cpu, use_disk, description in all_cache_configurations:
        cache = cache_config_factory(
            use_cpu_cache=use_cpu,
            use_disk_cache=use_disk
        )
        
        # Store data
        cache.put(0, sample_datapoint)
        
        # Verify sizes match expectations
        expected_cpu_size = 1 if use_cpu else 0
        expected_disk_size = 1 if use_disk else 0
        
        assert cache.get_cpu_size() == expected_cpu_size, f"CPU size mismatch for {description}"
        assert cache.get_disk_size() == expected_disk_size, f"Disk size mismatch for {description}"


def test_cache_exists_on_disk_functionality(cache_with_both_enabled, sample_datapoint):
    """Test disk existence checking functionality."""
    cache = cache_with_both_enabled
    
    # Initially, item should not exist on disk
    assert cache.exists_on_disk(0) is False
    
    # Store data - should now exist on disk
    cache.put(0, sample_datapoint)
    assert cache.exists_on_disk(0) is True
    
    # Clear CPU cache - should still exist on disk
    cache.clear_cpu_cache()
    assert cache.exists_on_disk(0) is True
    
    # Clear disk cache - should no longer exist on disk
    cache.clear_disk_cache()
    assert cache.exists_on_disk(0) is False


def test_cache_exists_on_disk_with_disk_disabled(cache_cpu_only, sample_datapoint):
    """Test disk existence checking when disk cache is disabled."""
    cache = cache_cpu_only
    
    # Should always return False when disk cache is disabled
    assert cache.exists_on_disk(0) is False
    
    # Even after storing data (CPU only)
    cache.put(0, sample_datapoint)
    assert cache.exists_on_disk(0) is False


def test_cache_get_info_comprehensive(cache_with_both_enabled, sample_datapoint):
    """Test comprehensive cache information retrieval."""
    cache = cache_with_both_enabled
    
    # Get info for empty cache
    info = cache.get_info()
    
    # Verify basic configuration info
    assert info['use_cpu_cache'] is True
    assert info['use_disk_cache'] is True
    assert info['cpu_cache_size'] == 0
    assert info['disk_cache_size'] == 0
    assert 'cache_dir' in info
    assert 'version_hash' in info
    
    # Store some data and verify info updates
    cache.put(0, sample_datapoint)
    cache.put(1, sample_datapoint)
    
    info_updated = cache.get_info()
    assert info_updated['cpu_cache_size'] == 2
    assert info_updated['disk_cache_size'] == 2
    
    # Verify cache directory and version info
    expected_cache_dir = f"{cache.disk_cache.cache_dir}"
    assert info_updated['cache_dir'] == expected_cache_dir
    assert info_updated['version_hash'] == cache.disk_cache.version_hash


def test_cache_get_info_with_different_configurations(all_cache_configurations, cache_config_factory):
    """Test get_info with different cache configurations."""
    for use_cpu, use_disk, description in all_cache_configurations:
        cache = cache_config_factory(
            use_cpu_cache=use_cpu,
            use_disk_cache=use_disk
        )
        
        info = cache.get_info()
        
        # Verify configuration is correctly reported
        assert info['use_cpu_cache'] == use_cpu, f"CPU config mismatch in info for {description}"
        assert info['use_disk_cache'] == use_disk, f"Disk config mismatch in info for {description}"
        
        # Verify size reporting
        assert info['cpu_cache_size'] == 0
        assert info['disk_cache_size'] == 0
        
        # Verify disk-specific info presence
        if use_disk:
            assert 'cache_dir' in info
            assert 'version_hash' in info
        else:
            # When disk cache is disabled, these might still be present but should reflect the disabled state
            pass


def test_cache_info_after_operations(cache_with_both_enabled, make_datapoint_factory):
    """Test that cache info correctly reflects state after various operations."""
    cache = cache_with_both_enabled
    
    # Add multiple items
    for i in range(3):
        cache.put(i, make_datapoint_factory(i))
    
    info = cache.get_info()
    assert info['cpu_cache_size'] == 3
    assert info['disk_cache_size'] == 3
    
    # Clear CPU cache
    cache.clear_cpu_cache()
    info_after_cpu_clear = cache.get_info()
    assert info_after_cpu_clear['cpu_cache_size'] == 0
    assert info_after_cpu_clear['disk_cache_size'] == 3
    
    # Promote one item back
    cache.get(0)
    info_after_promotion = cache.get_info()
    assert info_after_promotion['cpu_cache_size'] == 1
    assert info_after_promotion['disk_cache_size'] == 3
    
    # Clear all
    cache.clear_all()
    info_after_clear_all = cache.get_info()
    assert info_after_clear_all['cpu_cache_size'] == 0
    assert info_after_clear_all['disk_cache_size'] == 0


def test_cache_multiple_exists_checks(cache_with_both_enabled, make_datapoint_factory):
    """Test multiple exists_on_disk checks with various items."""
    cache = cache_with_both_enabled
    
    # Check non-existent items
    for i in range(5):
        assert cache.exists_on_disk(i) is False
    
    # Store some items
    for i in range(0, 5, 2):  # Store items 0, 2, 4
        cache.put(i, make_datapoint_factory(i))
    
    # Check existence pattern
    for i in range(5):
        expected_exists = i in [0, 2, 4]
        assert cache.exists_on_disk(i) == expected_exists, f"Existence check failed for item {i}"


def test_cache_clear_cpu_cache_recreation(cache_with_both_enabled, sample_datapoint):
    """Test that clear_cpu_cache properly recreates the CPU cache with same parameters."""
    cache = cache_with_both_enabled
    
    # Store original CPU cache parameters
    original_max_memory = cache.cpu_cache.max_memory_percent
    original_validation = cache.cpu_cache.enable_validation
    
    # Store some data
    cache.put(0, sample_datapoint)
    
    # Clear CPU cache
    cache.clear_cpu_cache()
    
    # Verify CPU cache was recreated with same parameters
    assert cache.cpu_cache is not None
    assert cache.cpu_cache.max_memory_percent == original_max_memory
    assert cache.cpu_cache.enable_validation == original_validation
    
    # Verify cache is indeed empty
    assert cache.get_cpu_size() == 0
    assert len(cache.cpu_cache.cache) == 0


def test_cache_disk_existence_with_manual_disk_operations(cache_with_both_enabled, sample_datapoint):
    """Test disk existence checking with manual disk cache operations."""
    cache = cache_with_both_enabled
    
    # Manually store in disk cache only
    cache.disk_cache.put(0, sample_datapoint)
    
    # Should exist on disk
    assert cache.exists_on_disk(0) is True
    
    # CPU cache should be empty
    assert cache.get_cpu_size() == 0
    
    # Combined cache get should find it and promote
    retrieved = cache.get(0)
    assert retrieved is not None
    
    # Should still exist on disk after promotion
    assert cache.exists_on_disk(0) is True
    assert cache.get_cpu_size() == 1


def test_cache_management_thread_safety_basic(cache_with_both_enabled, make_datapoint_factory):
    """Test basic thread safety of management operations."""
    import threading
    import time
    
    cache = cache_with_both_enabled
    
    # Store initial data
    for i in range(10):
        cache.put(i, make_datapoint_factory(i))
    
    results = []
    
    def clear_and_check():
        time.sleep(0.01)  # Small delay to increase chance of race conditions
        cache.clear_cpu_cache()
        results.append(cache.get_cpu_size())
    
    def get_info():
        time.sleep(0.01)
        info = cache.get_info()
        results.append(info['cpu_cache_size'])
    
    # Run operations concurrently
    threads = []
    for _ in range(5):
        t1 = threading.Thread(target=clear_and_check)
        t2 = threading.Thread(target=get_info)
        threads.extend([t1, t2])
        t1.start()
        t2.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # All operations should complete without crashing
    assert len(results) == 10