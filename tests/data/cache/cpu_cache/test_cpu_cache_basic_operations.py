import pytest
import torch
from data.cache.cpu_dataset_cache import CPUDatasetCache


def test_cache_put_and_get(sample_datapoint, cache_key_factory):
    """Test basic put and get operations."""
    cache = CPUDatasetCache()

    key0 = cache_key_factory(0)
    key1 = cache_key_factory(1)

    # Test put
    cache.put(sample_datapoint, cache_filepath=key0)
    assert len(cache.cache) == 1

    # Test get
    retrieved = cache.get(cache_filepath=key0)
    assert retrieved is not None
    assert retrieved['inputs']['image'].shape == sample_datapoint['inputs']['image'].shape
    assert torch.all(retrieved['inputs']['image'] == sample_datapoint['inputs']['image'])
    assert retrieved['labels']['class'] == sample_datapoint['labels']['class']
    assert retrieved['meta_info']['filename'] == sample_datapoint['meta_info']['filename']

    # Test get with non-existent key
    assert cache.get(cache_filepath=key1) is None


def test_cache_deep_copy_isolation(sample_datapoint, cache_key_factory):
    """Test that cached items are properly isolated through deep copying."""
    cache = CPUDatasetCache()
    key0 = cache_key_factory(0)

    # Test 1: Modifying original data
    cache.put(sample_datapoint, cache_filepath=key0)
    sample_datapoint['inputs']['image'] += 1.0
    sample_datapoint['meta_info']['filename'] = 'modified.jpg'

    cached = cache.get(cache_filepath=key0)
    assert not torch.all(cached['inputs']['image'] == sample_datapoint['inputs']['image'])
    assert cached['meta_info']['filename'] == 'test.jpg'

    # Test 2: Modifying retrieved data
    retrieved = cache.get(cache_filepath=key0)
    retrieved['inputs']['image'] += 1.0
    retrieved['meta_info']['filename'] = 'modified2.jpg'

    cached_again = cache.get(cache_filepath=key0)
    assert not torch.all(cached_again['inputs']['image'] == retrieved['inputs']['image'])
    assert cached_again['meta_info']['filename'] == 'test.jpg'


def test_cache_validation_data_corruption(sample_datapoint, cache_key_factory):
    """Test cache validation mechanism when cached data is corrupted."""
    cache = CPUDatasetCache(enable_validation=True)
    key0 = cache_key_factory(0)

    # Store data
    cache.put(sample_datapoint, cache_filepath=key0)
    assert cache.get(cache_filepath=key0) is not None

    # Corrupt the cached data (this simulates memory corruption or external modification)
    cache.cache[key0]['inputs']['image'] += 1.0  # Modify tensor in-place

    # Clear validated keys to force validation on next access
    cache.validated_keys.clear()

    with pytest.raises(ValueError, match=f"Cache validation failed for key {key0} - data corruption detected"):
        cache.get(cache_filepath=key0)  # Should raise ValueError due to data corruption


def test_cache_validation_checksum_corruption(sample_datapoint, cache_key_factory):
    """Test cache validation mechanism when stored checksum is corrupted."""
    cache = CPUDatasetCache(enable_validation=True)
    key0 = cache_key_factory(0)

    # Store data
    cache.put(sample_datapoint, cache_filepath=key0)
    assert cache.get(cache_filepath=key0) is not None

    # Corrupt the stored checksum (this simulates checksum corruption)
    cache.checksums[key0] = "corrupted_checksum_that_wont_match"

    # Clear validated keys to force validation on next access
    cache.validated_keys.clear()

    with pytest.raises(ValueError, match=f"Cache validation failed for key {key0} - data corruption detected"):
        cache.get(cache_filepath=key0)  # Should raise ValueError due to checksum mismatch
