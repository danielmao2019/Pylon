import pytest
import torch
from data.cache import DatasetCache


def test_cache_put_and_get(sample_datapoint):
    """Test basic put and get operations."""
    cache = DatasetCache()

    # Test put
    cache.put(0, sample_datapoint)
    assert len(cache.cache) == 1

    # Test get
    retrieved = cache.get(0)
    assert retrieved is not None
    assert retrieved['inputs']['image'].shape == sample_datapoint['inputs']['image'].shape
    assert torch.all(retrieved['inputs']['image'] == sample_datapoint['inputs']['image'])
    assert retrieved['labels']['class'] == sample_datapoint['labels']['class']
    assert retrieved['meta_info']['filename'] == sample_datapoint['meta_info']['filename']

    # Test get with non-existent key
    assert cache.get(1) is None


def test_cache_deep_copy_isolation(sample_datapoint):
    """Test that cached items are properly isolated through deep copying."""
    cache = DatasetCache()

    # Test 1: Modifying original data
    cache.put(0, sample_datapoint)
    sample_datapoint['inputs']['image'] += 1.0
    sample_datapoint['meta_info']['filename'] = 'modified.jpg'

    cached = cache.get(0)
    assert not torch.all(cached['inputs']['image'] == sample_datapoint['inputs']['image'])
    assert cached['meta_info']['filename'] == 'test.jpg'

    # Test 2: Modifying retrieved data
    retrieved = cache.get(0)
    retrieved['inputs']['image'] += 1.0
    retrieved['meta_info']['filename'] = 'modified2.jpg'

    cached_again = cache.get(0)
    assert not torch.all(cached_again['inputs']['image'] == retrieved['inputs']['image'])
    assert cached_again['meta_info']['filename'] == 'test.jpg'


def test_cache_validation(sample_datapoint):
    """Test cache validation mechanism."""
    cache = DatasetCache(enable_validation=True)

    # Test normal validation
    cache.put(0, sample_datapoint)
    assert cache.get(0) is not None

    # Test validation failure by corrupting cached data
    cache.cache[0]['inputs']['image'] += 1.0  # Modify tensor in-place
    with pytest.raises(ValueError, match="Cache validation failed for key 0 - data corruption detected"):
        cache.get(0)  # Should raise ValueError
    assert cache.validation_failures == 1
