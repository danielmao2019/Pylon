import pytest
import torch
import psutil
from data.cache import DatasetCache


@pytest.fixture
def sample_tensor():
    return torch.randn(3, 64, 64)


@pytest.fixture
def sample_datapoint(sample_tensor):
    return {
        'inputs': {'image': sample_tensor},
        'labels': {'class': torch.tensor([1])},
        'meta_info': {'filename': 'test.jpg'}
    }


@pytest.fixture
def cache_with_items(sample_datapoint):
    """Create a cache with initial items."""
    cache = DatasetCache()
    for i in range(3):
        cache.put(i, sample_datapoint)
    return cache


@pytest.mark.parametrize("scenario,access_order,expected_evicted,expected_retained", [
    (
        "basic_lru",  # Basic LRU eviction
        [0, 2, 1, 2, 0],  # Access order
        [1],  # Expected evicted items
        [0, 2, 3]  # Expected retained items
    ),
    (
        "no_access",  # No access after initial put
        [],  # No access
        [0],  # First item should be evicted
        [1, 2, 3]  # Later items retained
    ),
])
def test_lru_eviction_scenarios(cache_with_items, sample_datapoint, 
                              scenario, access_order, expected_evicted, expected_retained):
    """Test different LRU eviction scenarios."""
    cache = cache_with_items
    
    # Access items in specified order
    for i in access_order:
        cache.get(i)
    
    # Force eviction by setting low memory limit
    cache.max_memory_percent = psutil.Process().memory_percent() - 0.1
    
    # Add new item to trigger eviction
    cache.put(3, sample_datapoint)
    
    # Verify evicted items
    for item in expected_evicted:
        assert item not in cache.cache, f"Item {item} should have been evicted"
    
    # Verify retained items
    for item in expected_retained:
        assert item in cache.cache, f"Item {item} should have been retained"


@pytest.mark.parametrize("scenario,setup_actions,verify_actions,expected_evicted,expected_retained", [
    (
        "reput_updates_lru",  # Re-putting items updates LRU order
        [
            ("put", 0),  # Re-put item 0
        ],
        [
            ("put", 3),  # Add new item to force eviction
        ],
        [1],  # Item 1 should be evicted
        [0, 2, 3]  # Items 0, 2, 3 should be retained
    ),
    (
        "multiple_evictions",  # Multiple evictions maintain order
        [
            ("put", i) for i in range(5)  # Put 5 items
        ] + [
            ("get", i) for i in [2, 0, 4, 1, 3]  # Access in specific order
        ],
        [
            ("put", 5),  # Add two new items
            ("put", 6),
        ],
        [2, 0],  # First two accessed should be evicted
        [4, 1, 3, 5, 6]  # Rest should be retained
    ),
])
def test_lru_complex_scenarios(sample_datapoint, scenario, setup_actions, 
                             verify_actions, expected_evicted, expected_retained):
    """Test complex LRU scenarios with multiple operations."""
    cache = DatasetCache()
    
    # Perform setup actions
    for action, key in setup_actions:
        if action == "put":
            cache.put(key, sample_datapoint)
        elif action == "get":
            cache.get(key)
    
    # Force eviction by setting low memory limit
    cache.max_memory_percent = psutil.Process().memory_percent() - 0.1
    
    # Perform verification actions
    for action, key in verify_actions:
        if action == "put":
            cache.put(key, sample_datapoint)
        elif action == "get":
            cache.get(key)
    
    # Verify evicted items
    for item in expected_evicted:
        assert item not in cache.cache, f"Item {item} should have been evicted"
    
    # Verify retained items
    for item in expected_retained:
        assert item in cache.cache, f"Item {item} should have been retained"
