"""Test combined cache hierarchical behavior and cache promotion."""

import pytest
import torch


def test_cache_hierarchy_cpu_priority(cache_with_both_enabled, sample_datapoint, different_datapoint):
    """Test that CPU cache has priority over disk cache in retrieval."""
    cache = cache_with_both_enabled
    
    # Store different data in each cache layer
    cache.cpu_cache.put(0, sample_datapoint)
    cache.disk_cache.put(0, different_datapoint)
    
    # Get should return CPU version (higher priority)
    retrieved = cache.get(0)
    assert retrieved is not None
    assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])
    assert retrieved['labels']['class_id'] == sample_datapoint['labels']['class_id']
    assert retrieved['meta_info']['path'] == sample_datapoint['meta_info']['path']
    
    # Should NOT return disk version
    assert not torch.allclose(retrieved['inputs']['image'], different_datapoint['inputs']['image'])


def test_cache_hierarchy_disk_fallback(cache_with_both_enabled, sample_datapoint):
    """Test that disk cache is used when CPU cache misses."""
    cache = cache_with_both_enabled
    
    # Store data only in disk cache
    cache.disk_cache.put(0, sample_datapoint)
    
    # Verify CPU cache is empty but disk cache has data
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 1
    assert cache.cpu_cache.get(0) is None
    assert cache.disk_cache.get(0) is not None
    
    # Get should return disk data
    retrieved = cache.get(0)
    assert retrieved is not None
    assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])


def test_cache_promotion_disk_to_cpu(cache_with_both_enabled, sample_datapoint):
    """Test that disk cache hits are automatically promoted to CPU cache."""
    cache = cache_with_both_enabled
    
    # Store data only in disk cache
    cache.disk_cache.put(0, sample_datapoint)
    
    # Verify initial state: disk has data, CPU is empty
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 1
    
    # Get data - should trigger promotion
    retrieved = cache.get(0)
    assert retrieved is not None
    
    # Verify promotion occurred: data now in both caches
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 1
    
    # Verify CPU cache now contains the data
    cpu_data = cache.cpu_cache.get(0)
    assert cpu_data is not None
    assert torch.allclose(cpu_data['inputs']['image'], sample_datapoint['inputs']['image'])


def test_cache_no_promotion_when_cpu_disabled(cache_disk_only, sample_datapoint):
    """Test that promotion doesn't occur when CPU cache is disabled."""
    cache = cache_disk_only
    
    # Store data in disk cache
    cache.put(0, sample_datapoint)
    
    # Verify data is retrieved correctly
    retrieved = cache.get(0)
    assert retrieved is not None
    assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])
    
    # Verify no CPU cache exists (can't promote)
    assert cache.cpu_cache is None
    assert cache.get_cpu_size() == 0


def test_cache_hierarchy_miss_both_caches(cache_with_both_enabled):
    """Test behavior when item is not found in either cache."""
    cache = cache_with_both_enabled
    
    # Try to get non-existent item
    retrieved = cache.get(999)
    assert retrieved is None
    
    # Verify no changes to cache sizes
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 0


def test_cache_hierarchy_with_device_parameter(cache_with_both_enabled, sample_datapoint):
    """Test hierarchical retrieval with device parameter propagation."""
    cache = cache_with_both_enabled
    
    # Store data only in disk cache
    cache.disk_cache.put(0, sample_datapoint)
    
    # Get with specific device should promote to CPU
    device = 'cpu'
    retrieved = cache.get(0, device=device)
    assert retrieved is not None
    
    # Verify promotion occurred
    assert cache.get_cpu_size() == 1
    
    # Note: Device parameter affects disk cache loading,
    # but CPU cache promotion always stores on CPU


def test_cache_hierarchy_multiple_promotions(cache_with_both_enabled, make_datapoint_factory):
    """Test multiple disk-to-CPU promotions work correctly."""
    cache = cache_with_both_enabled
    datapoints = [make_datapoint_factory(i) for i in range(3)]
    
    # Store all data only in disk cache
    for i, datapoint in enumerate(datapoints):
        cache.disk_cache.put(i, datapoint)
    
    # Verify initial state
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == 3
    
    # Retrieve items one by one - each should trigger promotion
    for i, original_datapoint in enumerate(datapoints):
        retrieved = cache.get(i)
        assert retrieved is not None
        assert torch.allclose(retrieved['inputs']['image'], original_datapoint['inputs']['image'])
        
        # Verify promotion count increases
        assert cache.get_cpu_size() == i + 1
        assert cache.get_disk_size() == 3  # Disk size remains constant


def test_cache_hierarchy_partial_overlap(cache_with_both_enabled, make_datapoint_factory):
    """Test hierarchy behavior with partial overlap between CPU and disk caches."""
    cache = cache_with_both_enabled
    datapoints = [make_datapoint_factory(i) for i in range(5)]
    
    # Store some items in both caches, some only in disk
    for i in range(2):  # Items 0, 1 in both caches
        cache.put(i, datapoints[i])
    
    for i in range(2, 5):  # Items 2, 3, 4 only in disk cache
        cache.disk_cache.put(i, datapoints[i])
    
    # Verify initial state
    assert cache.get_cpu_size() == 2
    assert cache.get_disk_size() == 5
    
    # Retrieve all items
    for i, original_datapoint in enumerate(datapoints):
        retrieved = cache.get(i)
        assert retrieved is not None
        assert torch.allclose(retrieved['inputs']['image'], original_datapoint['inputs']['image'])
    
    # Verify final state: all items promoted to CPU
    assert cache.get_cpu_size() == 5
    assert cache.get_disk_size() == 5


def test_cache_hierarchy_promotion_preserves_data_integrity(cache_with_both_enabled, sample_datapoint):
    """Test that cache promotion preserves data integrity completely."""
    cache = cache_with_both_enabled
    
    # Store complex datapoint in disk cache
    cache.disk_cache.put(0, sample_datapoint)
    
    # Get data to trigger promotion
    retrieved = cache.get(0)
    
    # Get again - should come from CPU cache now
    retrieved_again = cache.get(0)
    
    # Verify both retrievals are identical
    assert torch.allclose(retrieved['inputs']['image'], retrieved_again['inputs']['image'])
    assert torch.equal(retrieved['labels']['mask'], retrieved_again['labels']['mask'])
    assert retrieved['meta_info'] == retrieved_again['meta_info']
    
    # Verify against original
    assert torch.allclose(retrieved_again['inputs']['image'], sample_datapoint['inputs']['image'])


def test_cache_hierarchy_with_cache_configuration_changes(hierarchical_test_setup, sample_datapoint):
    """Test hierarchy behavior with different cache configuration scenarios."""
    # Test CPU-only scenario (no promotion possible)
    cache = hierarchical_test_setup()
    cache.disk_cache = None  # Simulate disk cache disabled
    cache.use_disk_cache = False
    
    # Store in CPU cache only
    cache.cpu_cache.put(0, sample_datapoint)
    
    retrieved = cache.get(0)
    assert retrieved is not None
    assert torch.allclose(retrieved['inputs']['image'], sample_datapoint['inputs']['image'])
    assert cache.get_cpu_size() == 1
    assert cache.get_disk_size() == 0


def test_cache_hierarchy_promotion_isolation(cache_with_both_enabled, sample_datapoint):
    """Test that promotion creates isolated copies, not references."""
    cache = cache_with_both_enabled
    
    # Store data in disk cache
    cache.disk_cache.put(0, sample_datapoint)
    
    # Get to trigger promotion
    retrieved_first = cache.get(0)
    
    # Modify the retrieved data
    retrieved_first['inputs']['image'] += 100.0
    retrieved_first['meta_info']['modified'] = True
    
    # Get again - should be unaffected by modifications
    retrieved_second = cache.get(0)
    assert not torch.allclose(retrieved_second['inputs']['image'], retrieved_first['inputs']['image'])
    assert 'modified' not in retrieved_second['meta_info']
    
    # Should match original
    assert torch.allclose(retrieved_second['inputs']['image'], sample_datapoint['inputs']['image'])


def test_cache_hierarchy_stress_promotion(cache_with_both_enabled, make_datapoint_factory):
    """Test cache hierarchy under stress with many promotions."""
    cache = cache_with_both_enabled
    num_items = 20
    datapoints = [make_datapoint_factory(i) for i in range(num_items)]
    
    # Store all items only in disk cache
    for i, datapoint in enumerate(datapoints):
        cache.disk_cache.put(i, datapoint)
    
    # Verify initial state
    assert cache.get_cpu_size() == 0
    assert cache.get_disk_size() == num_items
    
    # Retrieve all items in random order to test promotion
    import random
    indices = list(range(num_items))
    random.shuffle(indices)
    
    for idx in indices:
        retrieved = cache.get(idx)
        assert retrieved is not None
        assert torch.allclose(retrieved['inputs']['image'], datapoints[idx]['inputs']['image'])
    
    # Verify all items were promoted
    assert cache.get_cpu_size() == num_items
    assert cache.get_disk_size() == num_items
    
    # Verify subsequent retrievals come from CPU (fast path)
    for i in range(num_items):
        retrieved = cache.get(i)
        assert retrieved is not None
        # Should be very fast since coming from CPU cache now


def test_cache_hierarchy_promotion_with_different_device_requests(cache_with_both_enabled, sample_datapoint):
    """Test that device parameter works correctly during promotion."""
    cache = cache_with_both_enabled
    
    # Store data in disk cache
    cache.disk_cache.put(0, sample_datapoint)
    
    # First retrieval with specific device should promote correctly
    device = 'cpu'
    retrieved = cache.get(0, device=device)
    assert retrieved is not None
    
    # Verify promotion occurred
    assert cache.get_cpu_size() == 1
    
    # Second retrieval should come from CPU cache
    retrieved_again = cache.get(0, device=device)
    assert retrieved_again is not None
    
    # Verify data consistency regardless of device parameter
    assert torch.allclose(retrieved['inputs']['image'], retrieved_again['inputs']['image'])