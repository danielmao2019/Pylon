"""Test combined cache hierarchical behavior and cache promotion."""

import os

import pytest
import torch


def _fp(cache, idx: int) -> str:
    return os.path.join(cache.version_dir, f"{idx}.pt")


def _get(cache, idx: int, device=None):
    return cache.get(cache_filepath=_fp(cache, idx), device=device)


def _cpu_size(cache) -> int:
    cache._wait_for_pending_puts()
    if cache.cpu_cache is None:
        return 0
    return len(cache.cpu_cache.cache)


def _disk_size(cache) -> int:
    cache._wait_for_pending_puts()
    if cache.disk_cache is None:
        return 0
    return cache.disk_cache.get_size()


def test_cache_hierarchy_cpu_priority(
    cache_with_both_enabled, sample_datapoint, different_datapoint
):
    """Test that CPU cache has priority over disk cache in retrieval."""
    cache = cache_with_both_enabled
    cache_key = _fp(cache, 0)

    # Store different data in each cache layer
    cache.cpu_cache.put(sample_datapoint, cache_filepath=cache_key)
    cache.disk_cache.put(different_datapoint, cache_filepath=_fp(cache, 0))

    # Get should return CPU version (higher priority)
    retrieved = _get(cache, 0)
    assert retrieved is not None
    assert torch.allclose(
        retrieved['inputs']['image'], sample_datapoint['inputs']['image']
    )
    assert retrieved['labels']['class_id'] == sample_datapoint['labels']['class_id']
    assert retrieved['meta_info']['path'] == sample_datapoint['meta_info']['path']

    # Should NOT return disk version
    assert not torch.allclose(
        retrieved['inputs']['image'], different_datapoint['inputs']['image']
    )


def test_cache_hierarchy_disk_fallback(cache_with_both_enabled, sample_datapoint):
    """Test that disk cache is used when CPU cache misses."""
    cache = cache_with_both_enabled

    # Store data only in disk cache
    cache.disk_cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))

    # Verify CPU cache is empty but disk cache has data
    assert _cpu_size(cache) == 0
    assert _disk_size(cache) == 1
    assert cache.cpu_cache.get(cache_filepath=_fp(cache, 0)) is None
    assert cache.disk_cache.get(cache_filepath=_fp(cache, 0)) is not None

    # Get should return disk data
    retrieved = _get(cache, 0)
    assert retrieved is not None
    assert torch.allclose(
        retrieved['inputs']['image'], sample_datapoint['inputs']['image']
    )


def test_cache_promotion_disk_to_cpu(cache_with_both_enabled, sample_datapoint):
    """Test that disk cache hits are automatically promoted to CPU cache."""
    cache = cache_with_both_enabled

    # Store data only in disk cache
    cache.disk_cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))

    # Verify initial state: disk has data, CPU is empty
    assert _cpu_size(cache) == 0
    assert _disk_size(cache) == 1

    # Get data - should trigger promotion
    retrieved = _get(cache, 0)
    assert retrieved is not None

    # Verify promotion occurred: data now in both caches
    assert _cpu_size(cache) == 1
    assert _disk_size(cache) == 1

    # Verify CPU cache now contains the data
    cpu_data = cache.cpu_cache.get(cache_filepath=_fp(cache, 0))
    assert cpu_data is not None
    assert torch.allclose(
        cpu_data['inputs']['image'], sample_datapoint['inputs']['image']
    )


def test_cache_no_promotion_when_cpu_disabled(cache_disk_only, sample_datapoint):
    """Test that promotion doesn't occur when CPU cache is disabled."""
    cache = cache_disk_only

    # Store data in disk cache
    cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))

    # Verify data is retrieved correctly
    retrieved = _get(cache, 0)
    assert retrieved is not None
    assert torch.allclose(
        retrieved['inputs']['image'], sample_datapoint['inputs']['image']
    )

    # Verify no CPU cache exists (can't promote)
    assert cache.cpu_cache is None
    assert _cpu_size(cache) == 0


def test_cache_hierarchy_miss_both_caches(cache_with_both_enabled):
    """Test behavior when item is not found in either cache."""
    cache = cache_with_both_enabled

    # Try to get non-existent item
    retrieved = cache.get(cache_filepath=_fp(cache, 999))
    assert retrieved is None

    # Verify no changes to cache sizes
    assert _cpu_size(cache) == 0
    assert _disk_size(cache) == 0


def test_cache_hierarchy_with_device_parameter(
    cache_with_both_enabled, sample_datapoint
):
    """Test hierarchical retrieval with device parameter propagation."""
    cache = cache_with_both_enabled

    # Store data only in disk cache
    cache.disk_cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))

    # Get with specific device should promote to CPU
    device = 'cpu'
    retrieved = _get(cache, 0, device=device)
    assert retrieved is not None

    # Verify promotion occurred
    assert _cpu_size(cache) == 1

    # Note: Device parameter affects disk cache loading,
    # but CPU cache promotion always stores on CPU


def test_cache_hierarchy_multiple_promotions(
    cache_with_both_enabled, make_datapoint_factory
):
    """Test multiple disk-to-CPU promotions work correctly."""
    cache = cache_with_both_enabled
    datapoints = [make_datapoint_factory(i) for i in range(3)]

    # Store all data only in disk cache
    for i, datapoint in enumerate(datapoints):
        cache.disk_cache.put(datapoint, cache_filepath=_fp(cache, i))

    # Verify initial state
    assert _cpu_size(cache) == 0
    assert _disk_size(cache) == 3

    # Retrieve items one by one - each should trigger promotion
    for i, original_datapoint in enumerate(datapoints):
        retrieved = _get(cache, i)
        assert retrieved is not None
        assert torch.allclose(
            retrieved['inputs']['image'], original_datapoint['inputs']['image']
        )

        # Verify promotion count increases
        assert _cpu_size(cache) == i + 1
        assert _disk_size(cache) == 3  # Disk size remains constant


def test_cache_hierarchy_partial_overlap(
    cache_with_both_enabled, make_datapoint_factory
):
    """Test hierarchy behavior with partial overlap between CPU and disk caches."""
    cache = cache_with_both_enabled
    datapoints = [make_datapoint_factory(i) for i in range(5)]

    # Store some items in both caches, some only in disk
    for i in range(2):  # Items 0, 1 in both caches
        cache.put(datapoints[i], cache_filepath=_fp(cache, i))

    for i in range(2, 5):  # Items 2, 3, 4 only in disk cache
        cache.disk_cache.put(datapoints[i], cache_filepath=_fp(cache, i))

    # Verify initial state
    assert _cpu_size(cache) == 2
    assert _disk_size(cache) == 5

    # Retrieve all items
    for i, original_datapoint in enumerate(datapoints):
        retrieved = _get(cache, i)
        assert retrieved is not None
        assert torch.allclose(
            retrieved['inputs']['image'], original_datapoint['inputs']['image']
        )

    # Verify final state: all items promoted to CPU
    assert _cpu_size(cache) == 5
    assert _disk_size(cache) == 5


def test_cache_hierarchy_promotion_preserves_data_integrity(
    cache_with_both_enabled, sample_datapoint
):
    """Test that cache promotion preserves data integrity completely."""
    cache = cache_with_both_enabled

    # Store complex datapoint in disk cache
    cache.disk_cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))

    # Get data to trigger promotion
    retrieved = _get(cache, 0)

    # Get again - should come from CPU cache now
    retrieved_again = _get(cache, 0)

    # Verify both retrievals are identical
    assert torch.allclose(
        retrieved['inputs']['image'], retrieved_again['inputs']['image']
    )
    assert torch.equal(retrieved['labels']['mask'], retrieved_again['labels']['mask'])
    assert retrieved['meta_info'] == retrieved_again['meta_info']

    # Verify against original
    assert torch.allclose(
        retrieved_again['inputs']['image'], sample_datapoint['inputs']['image']
    )


def test_cache_hierarchy_with_cache_configuration_changes(
    hierarchical_test_setup, sample_datapoint
):
    """Test hierarchy behavior with different cache configuration scenarios."""
    # Test CPU-only scenario (no promotion possible)
    cache = hierarchical_test_setup()
    cache.disk_cache = None  # Simulate disk cache disabled
    cache.use_disk_cache = False

    # Store in CPU cache only
    cache.cpu_cache.put(
        sample_datapoint,
        cache_filepath=os.path.join(cache.cache_dir, "0.pt"),
    )

    retrieved = _get(cache, 0)
    assert retrieved is not None
    assert torch.allclose(
        retrieved['inputs']['image'], sample_datapoint['inputs']['image']
    )
    assert _cpu_size(cache) == 1
    assert _disk_size(cache) == 0


def test_cache_hierarchy_promotion_isolation(cache_with_both_enabled, sample_datapoint):
    """Test that promotion creates isolated copies, not references."""
    cache = cache_with_both_enabled

    # Store data in disk cache
    cache.disk_cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))

    # Get to trigger promotion
    retrieved_first = _get(cache, 0)

    # Modify the retrieved data
    retrieved_first['inputs']['image'] += 100.0
    retrieved_first['meta_info']['modified'] = True

    # Get again - should be unaffected by modifications
    retrieved_second = _get(cache, 0)
    assert not torch.allclose(
        retrieved_second['inputs']['image'], retrieved_first['inputs']['image']
    )
    assert 'modified' not in retrieved_second['meta_info']

    # Should match original
    assert torch.allclose(
        retrieved_second['inputs']['image'], sample_datapoint['inputs']['image']
    )


def test_cache_hierarchy_stress_promotion(
    cache_with_both_enabled, make_datapoint_factory
):
    """Test cache hierarchy under stress with many promotions."""
    cache = cache_with_both_enabled
    num_items = 20
    datapoints = [make_datapoint_factory(i) for i in range(num_items)]

    # Store all items only in disk cache
    for i, datapoint in enumerate(datapoints):
        cache.disk_cache.put(datapoint, cache_filepath=_fp(cache, i))

    # Verify initial state
    assert _cpu_size(cache) == 0
    assert _disk_size(cache) == num_items

    # Retrieve all items in random order to test promotion
    import random

    indices = list(range(num_items))
    random.shuffle(indices)

    for idx in indices:
        retrieved = _get(cache, idx)
        assert retrieved is not None
        assert torch.allclose(
            retrieved['inputs']['image'], datapoints[idx]['inputs']['image']
        )

    # Verify all items were promoted
    assert _cpu_size(cache) == num_items
    assert _disk_size(cache) == num_items

    # Verify subsequent retrievals come from CPU (fast path)
    for i in range(num_items):
        retrieved = _get(cache, i)
        assert retrieved is not None
        # Should be very fast since coming from CPU cache now


def test_cache_hierarchy_promotion_with_different_device_requests(
    cache_with_both_enabled, sample_datapoint
):
    """Test that device parameter works correctly during promotion."""
    cache = cache_with_both_enabled

    # Store data in disk cache
    cache.disk_cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))

    # First retrieval with specific device should promote correctly
    device = 'cpu'
    retrieved = _get(cache, 0, device=device)
    assert retrieved is not None

    # Verify promotion occurred
    assert _cpu_size(cache) == 1

    # Second retrieval should come from CPU cache
    retrieved_again = _get(cache, 0, device=device)
    assert retrieved_again is not None

    # Verify data consistency regardless of device parameter
    assert torch.allclose(
        retrieved['inputs']['image'], retrieved_again['inputs']['image']
    )
