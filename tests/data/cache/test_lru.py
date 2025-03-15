import pytest
import torch
import psutil
import time
from data.cache import DatasetCache


def calculate_datapoint_memory(datapoint):
    """Calculate approximate memory usage of a datapoint in bytes."""
    tensor = datapoint['inputs']['image']
    # Calculate tensor memory (float32 = 4 bytes)
    tensor_memory = tensor.numel() * 4  # tensor.element_size() could also be used
    # Add overhead for Python objects, metadata, etc (conservative estimate)
    overhead = 1024  # 1KB overhead for dict structure, metadata, etc
    return tensor_memory + overhead


def get_stable_memory_usage():
    """Get stable memory usage by taking multiple measurements."""
    measurements = []
    for _ in range(5):
        measurements.append(psutil.Process().memory_percent())
        time.sleep(0.1)  # Small delay between measurements
    return sum(measurements) / len(measurements)


@pytest.fixture
def cache_with_items(sample_datapoint):
    """Create a cache with initial items."""
    # Start with a low memory limit
    cache = DatasetCache(max_memory_percent=psutil.Process().memory_percent())
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
    
    print(f"\nTesting scenario: {scenario}")
    print(f"Initial cache state: {list(cache.cache.keys())}")
    
    # Access items in specified order
    for i in access_order:
        cache.get(i)
        print(f"After get({i}): {list(cache.cache.keys())}")
    
    # Add new item which should trigger eviction due to memory limit
    cache.put(3, sample_datapoint)
    print(f"After adding item 3: {list(cache.cache.keys())}")
    
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


def test_lru_order_verification(sample_datapoint):
    """Test that verifies the exact LRU ordering after each operation."""
    cache = DatasetCache()
    
    # Initial state
    for i in range(3):
        cache.put(i, sample_datapoint)
        print(f"After put({i}): {list(cache.cache.keys())}")
    assert list(cache.cache.keys()) == [0, 1, 2], "Initial order incorrect"
    
    # Test get updates order
    cache.get(0)  # Move 0 to end
    print(f"After get(0): {list(cache.cache.keys())}")
    assert list(cache.cache.keys()) == [1, 2, 0], "Order after get(0) incorrect"
    
    cache.get(1)  # Move 1 to end
    print(f"After get(1): {list(cache.cache.keys())}")
    assert list(cache.cache.keys()) == [2, 0, 1], "Order after get(1) incorrect"
    
    # Test put updates order
    cache.put(2, sample_datapoint)  # Re-put 2, should move to end
    print(f"After put(2): {list(cache.cache.keys())}")
    assert list(cache.cache.keys()) == [0, 1, 2], "Order after put(2) incorrect"
    
    # Test eviction removes oldest
    cache.max_memory_percent = psutil.Process().memory_percent() - 0.1
    print(f"Setting memory limit to {cache.max_memory_percent}%")
    cache.put(3, sample_datapoint)  # Should evict oldest (0)
    print(f"After put(3): {list(cache.cache.keys())}")
    assert list(cache.cache.keys()) == [1, 2, 3], "Order after eviction incorrect"
    
    # Test multiple operations
    cache.get(1)  # Move 1 to end
    print(f"After get(1): {list(cache.cache.keys())}")
    cache.put(4, sample_datapoint)  # Should evict oldest (2)
    print(f"After put(4): {list(cache.cache.keys())}")
    assert list(cache.cache.keys()) == [3, 1, 4], "Final order incorrect"
