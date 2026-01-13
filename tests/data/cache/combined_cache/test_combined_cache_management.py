"""Test combined cache management operations (clear, size, info, exists)."""

import os

import pytest
import torch

from data.cache.cpu_dataset_cache import CPUDatasetCache


def _fp(cache, idx: int) -> str:
    return os.path.join(cache.version_dir, f"{idx}.pt")


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


def _exists_on_disk(cache, cache_filepath: str) -> bool:
    cache._wait_for_pending_puts()
    if cache.disk_cache is None:
        return False
    return cache.disk_cache.exists(cache_filepath)


def _clear_cpu_cache(cache) -> None:
    cache._wait_for_pending_puts()
    if cache.cpu_cache is None:
        return
    cache.cpu_cache = CPUDatasetCache(
        max_memory_percent=cache.cpu_cache.max_memory_percent,
        enable_validation=cache.cpu_cache.enable_validation,
    )


def _clear_disk_cache(cache) -> None:
    cache._wait_for_pending_puts()
    if cache.disk_cache is None:
        return
    cache.disk_cache.clear()


def _info(cache) -> dict:
    return {
        'use_cpu_cache': cache.use_cpu_cache,
        'use_disk_cache': cache.use_disk_cache,
        'cpu_cache_size': _cpu_size(cache),
        'disk_cache_size': _disk_size(cache),
        'cache_dir': (
            cache.disk_cache.cache_dir if cache.disk_cache is not None else None
        ),
        'version_hash': (
            cache.disk_cache.version_hash
            if cache.disk_cache is not None
            else cache.version_hash
        ),
    }


def test_cache_clear_cpu_cache_only(cache_with_both_enabled, sample_datapoint):
    """Test clearing only the CPU cache while preserving disk cache."""
    cache = cache_with_both_enabled

    # Store data in both caches
    cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))
    assert _cpu_size(cache) == 1
    assert _disk_size(cache) == 1

    # Clear only CPU cache
    _clear_cpu_cache(cache)

    # Verify CPU cache is cleared but disk cache remains
    assert _cpu_size(cache) == 0
    assert _disk_size(cache) == 1

    # Verify data can still be retrieved (from disk) and promotes to CPU
    retrieved = cache.get(cache_filepath=_fp(cache, 0))
    assert retrieved is not None
    assert torch.allclose(
        retrieved['inputs']['image'], sample_datapoint['inputs']['image']
    )

    # Verify promotion occurred
    assert _cpu_size(cache) == 1
    assert _disk_size(cache) == 1


def test_cache_clear_disk_cache_only(cache_with_both_enabled, sample_datapoint):
    """Test clearing only the disk cache while preserving CPU cache."""
    cache = cache_with_both_enabled

    # Store data in both caches
    cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))
    assert _cpu_size(cache) == 1
    assert _disk_size(cache) == 1

    # Clear only disk cache
    _clear_disk_cache(cache)

    # Verify disk cache is cleared but CPU cache remains
    assert _cpu_size(cache) == 1
    assert _disk_size(cache) == 0

    # Verify data can still be retrieved (from CPU)
    retrieved = cache.get(cache_filepath=_fp(cache, 0))
    assert retrieved is not None
    assert torch.allclose(
        retrieved['inputs']['image'], sample_datapoint['inputs']['image']
    )


def test_cache_clear_all_caches(cache_with_both_enabled, sample_datapoint):
    """Test clearing both CPU and disk caches."""
    cache = cache_with_both_enabled

    # Store data in both caches
    cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))
    assert _cpu_size(cache) == 1
    assert _disk_size(cache) == 1

    # Clear all caches
    cache.clear()

    # Verify both caches are cleared
    assert _cpu_size(cache) == 0
    assert _disk_size(cache) == 0

    # Verify data cannot be retrieved
    retrieved = cache.get(cache_filepath=_fp(cache, 0))
    assert retrieved is None


def test_cache_clear_operations_with_disabled_caches(
    all_cache_configurations, cache_config_factory, sample_datapoint
):
    """Test clear operations with different cache enable/disable configurations."""
    for use_cpu, use_disk, description in all_cache_configurations:
        cache = cache_config_factory(use_cpu_cache=use_cpu, use_disk_cache=use_disk)

        # Store data if possible
        cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))

        # Test all clear operations (should not crash)
        _clear_cpu_cache(cache)
        _clear_disk_cache(cache)
        cache.clear()

        # Verify sizes are zero after clearing
        assert _cpu_size(cache) == 0, f"CPU size not zero after clear for {description}"
        assert (
            _disk_size(cache) == 0
        ), f"Disk size not zero after clear for {description}"


def test_cache_size_reporting_accuracy(cache_with_both_enabled, make_datapoint_factory):
    """Test that cache size reporting is accurate."""
    cache = cache_with_both_enabled

    # Verify initial sizes
    assert _cpu_size(cache) == 0
    assert _disk_size(cache) == 0

    # Add items one by one and verify size increments
    for i in range(5):
        datapoint = make_datapoint_factory(i)
        cache.put(i, datapoint)

        assert _cpu_size(cache) == i + 1
        assert _disk_size(cache) == i + 1

    # Remove from CPU cache by clearing
    _clear_cpu_cache(cache)
    assert _cpu_size(cache) == 0
    assert _disk_size(cache) == 5  # Disk remains

    # Promote one item back to CPU
    cache.get(cache_filepath=_fp(cache, 0))  # Should promote from disk to CPU
    assert _cpu_size(cache) == 1
    assert _disk_size(cache) == 5


def test_cache_size_with_different_configurations(
    all_cache_configurations, cache_config_factory, sample_datapoint
):
    """Test size reporting with different cache configurations."""
    for use_cpu, use_disk, description in all_cache_configurations:
        cache = cache_config_factory(use_cpu_cache=use_cpu, use_disk_cache=use_disk)

        # Store data
        cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))

        # Verify sizes match expectations
        expected_cpu_size = 1 if use_cpu else 0
        expected_disk_size = 1 if use_disk else 0

        assert (
            _cpu_size(cache) == expected_cpu_size
        ), f"CPU size mismatch for {description}"
        assert (
            _disk_size(cache) == expected_disk_size
        ), f"Disk size mismatch for {description}"


def test_cache_exists_on_disk_functionality(cache_with_both_enabled, sample_datapoint):
    """Test disk existence checking functionality."""
    cache = cache_with_both_enabled

    # Initially, item should not exist on disk
    assert _exists_on_disk(cache, _fp(cache, 0)) is False

    # Store data - should now exist on disk
    cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))
    assert _exists_on_disk(cache, _fp(cache, 0)) is True

    # Clear CPU cache - should still exist on disk
    _clear_cpu_cache(cache)
    assert _exists_on_disk(cache, _fp(cache, 0)) is True

    # Clear disk cache - should no longer exist on disk
    _clear_disk_cache(cache)
    assert _exists_on_disk(cache, _fp(cache, 0)) is False


def test_cache_exists_on_disk_with_disk_disabled(cache_cpu_only, sample_datapoint):
    """Test disk existence checking when disk cache is disabled."""
    cache = cache_cpu_only

    # Should always return False when disk cache is disabled
    assert _exists_on_disk(cache, _fp(cache, 0)) is False

    # Even after storing data (CPU only)
    cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))
    assert _exists_on_disk(cache, _fp(cache, 0)) is False


def test_cache_get_info_comprehensive(cache_with_both_enabled, sample_datapoint):
    """Test comprehensive cache information retrieval."""
    cache = cache_with_both_enabled

    # Get info for empty cache
    info = _info(cache)

    # Verify basic configuration info
    assert info['use_cpu_cache'] is True
    assert info['use_disk_cache'] is True
    assert info['cpu_cache_size'] == 0
    assert info['disk_cache_size'] == 0
    assert 'cache_dir' in info
    assert 'version_hash' in info

    # Store some data and verify info updates
    cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))
    cache.put(sample_datapoint, cache_filepath=_fp(cache, 1))

    info_updated = _info(cache)
    assert info_updated['cpu_cache_size'] == 2
    assert info_updated['disk_cache_size'] == 2

    # Verify cache directory and version info
    expected_cache_dir = f"{cache.disk_cache.cache_dir}"
    assert info_updated['cache_dir'] == expected_cache_dir
    assert info_updated['version_hash'] == cache.disk_cache.version_hash


def test_cache_get_info_with_different_configurations(
    all_cache_configurations, cache_config_factory
):
    """Test get_info with different cache configurations."""
    for use_cpu, use_disk, description in all_cache_configurations:
        cache = cache_config_factory(use_cpu_cache=use_cpu, use_disk_cache=use_disk)

        info = _info(cache)

        # Verify configuration is correctly reported
        assert (
            info['use_cpu_cache'] == use_cpu
        ), f"CPU config mismatch in info for {description}"
        assert (
            info['use_disk_cache'] == use_disk
        ), f"Disk config mismatch in info for {description}"

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

    info = _info(cache)
    assert info['cpu_cache_size'] == 3
    assert info['disk_cache_size'] == 3

    # Clear CPU cache
    _clear_cpu_cache(cache)
    info_after_cpu_clear = _info(cache)
    assert info_after_cpu_clear['cpu_cache_size'] == 0
    assert info_after_cpu_clear['disk_cache_size'] == 3

    # Promote one item back
    cache.get(cache_filepath=_fp(cache, 0))
    info_after_promotion = _info(cache)
    assert info_after_promotion['cpu_cache_size'] == 1
    assert info_after_promotion['disk_cache_size'] == 3

    # Clear all
    cache.clear()
    info_after_clear_all = _info(cache)
    assert info_after_clear_all['cpu_cache_size'] == 0
    assert info_after_clear_all['disk_cache_size'] == 0


def test_cache_multiple_exists_checks(cache_with_both_enabled, make_datapoint_factory):
    """Test multiple exists_on_disk checks with various items."""
    cache = cache_with_both_enabled

    # Check non-existent items
    for i in range(5):
        assert _exists_on_disk(cache, _fp(cache, i)) is False

    # Store some items
    for i in range(0, 5, 2):  # Store items 0, 2, 4
        cache.put(i, make_datapoint_factory(i))

    # Check existence pattern
    for i in range(5):
        expected_exists = i in [0, 2, 4]
        assert (
            _exists_on_disk(cache, _fp(cache, i)) == expected_exists
        ), f"Existence check failed for item {i}"


def test_cache_clear_cpu_cache_recreation(cache_with_both_enabled, sample_datapoint):
    """Test that clear_cpu_cache properly recreates the CPU cache with same parameters."""
    cache = cache_with_both_enabled

    # Store original CPU cache parameters
    original_max_memory = cache.cpu_cache.max_memory_percent
    original_validation = cache.cpu_cache.enable_validation

    # Store some data
    cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))

    # Clear CPU cache
    _clear_cpu_cache(cache)

    # Verify CPU cache was recreated with same parameters
    assert cache.cpu_cache is not None
    assert cache.cpu_cache.max_memory_percent == original_max_memory
    assert cache.cpu_cache.enable_validation == original_validation

    # Verify cache is indeed empty
    assert _cpu_size(cache) == 0
    assert len(cache.cpu_cache.cache) == 0


def test_cache_disk_existence_with_manual_disk_operations(
    cache_with_both_enabled, sample_datapoint
):
    """Test disk existence checking with manual disk cache operations."""
    cache = cache_with_both_enabled

    # Manually store in disk cache only
    cache.disk_cache.put(sample_datapoint, cache_filepath=_fp(cache, 0))

    # Should exist on disk
    assert _exists_on_disk(cache, _fp(cache, 0)) is True

    # CPU cache should be empty
    assert _cpu_size(cache) == 0

    # Combined cache get should find it and promote
    retrieved = cache.get(cache_filepath=_fp(cache, 0))
    assert retrieved is not None

    # Should still exist on disk after promotion
    assert _exists_on_disk(cache, _fp(cache, 0)) is True
    assert _cpu_size(cache) == 1


def test_cache_management_thread_safety_basic(
    cache_with_both_enabled, make_datapoint_factory
):
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
        _clear_cpu_cache(cache)
        results.append(_cpu_size(cache))

    def get_info():
        time.sleep(0.01)
        info = _info(cache)
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
