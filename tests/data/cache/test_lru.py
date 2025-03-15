import pytest
import torch
import psutil
from data.cache import DatasetCache


def calculate_datapoint_memory(datapoint):
    """Calculate approximate memory usage of a datapoint in bytes."""
    tensor = datapoint['inputs']['image']
    # Calculate tensor memory (float32 = 4 bytes)
    tensor_memory = tensor.numel() * 4
    # Add overhead for Python objects, metadata, etc
    overhead = 1024  # 1KB overhead
    return tensor_memory + overhead


@pytest.fixture
def cache_with_items(sample_datapoint):
    """Create a cache with initial items."""
    # Calculate memory needed for 3 items plus overhead
    item_memory = calculate_datapoint_memory(sample_datapoint)
    total_memory_needed = item_memory * 3
    total_memory_with_overhead = total_memory_needed + 1024  # 1KB overhead
    
    # Set percentage to allow exactly 3 items plus overhead
    max_percent = (total_memory_with_overhead / psutil.virtual_memory().total) * 100
    
    cache = DatasetCache(max_memory_percent=max_percent)
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
def test_lru_eviction_scenarios(
    cache_with_items, sample_datapoint, 
    scenario, access_order, expected_evicted, expected_retained,
):
    """Test different LRU eviction scenarios."""
    cache = cache_with_items
    
    # Access items in specified order
    for i in access_order:
        cache.get(i)
    
    # Add new item which should trigger eviction
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
            ("put", i) for i in range(3)  # Put 3 items
        ] + [
            ("get", i) for i in [2, 0, 1]  # Access in specific order
        ],
        [
            ("put", 3),  # Add new item
        ],
        [2],  # First accessed should be evicted
        [0, 1, 3]  # Rest should be retained
    ),
])
def test_lru_complex_scenarios(sample_datapoint, scenario, setup_actions, 
                             verify_actions, expected_evicted, expected_retained):
    """Test complex LRU scenarios with multiple operations."""
    # Calculate memory needed for 3 items plus overhead
    item_memory = calculate_datapoint_memory(sample_datapoint)
    total_memory_needed = item_memory * 3
    total_memory_with_overhead = total_memory_needed + 1024  # 1KB overhead
    
    # Set percentage to allow exactly 3 items plus overhead
    max_percent = (total_memory_with_overhead / psutil.virtual_memory().total) * 100
    
    cache = DatasetCache(max_memory_percent=max_percent)
    
    # Perform setup actions
    for action, key in setup_actions:
        if action == "put":
            cache.put(key, sample_datapoint)
        elif action == "get":
            cache.get(key)
    
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
