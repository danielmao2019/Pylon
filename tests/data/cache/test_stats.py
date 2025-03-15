import pytest
import torch
from data.cache import DatasetCache


def test_cache_stats(sample_datapoint):
    """Test cache statistics."""
    cache = DatasetCache()
    
    # Initial stats
    stats = cache.get_stats()
    assert stats['size'] == 0
    assert stats['hits'] == 0
    assert stats['misses'] == 0
    assert stats['hit_rate'] == 0
    
    # Add and access items
    cache.put(0, sample_datapoint)
    cache.get(0)  # Hit
    cache.get(1)  # Miss
    
    stats = cache.get_stats()
    assert stats['size'] == 1
    assert stats['hits'] == 1
    assert stats['misses'] == 1
    assert abs(stats['hit_rate'] - 0.5) < 1.0e-05


@pytest.mark.parametrize("scenario,actions,expected_stats", [
    (
        "all_hits",
        [
            ("put", 0),
            ("get", 0),
            ("get", 0),
            ("get", 0),
        ],
        {"hits": 3, "misses": 0, "hit_rate": 1.0, "size": 1}
    ),
    (
        "all_misses",
        [
            ("get", 0),
            ("get", 1),
            ("get", 2),
        ],
        {"hits": 0, "misses": 3, "hit_rate": 0.0, "size": 0}
    ),
    (
        "mixed_operations",
        [
            ("put", 0),
            ("get", 0),
            ("get", 1),
            ("put", 1),
            ("get", 1),
            ("get", 2),
        ],
        {"hits": 2, "misses": 2, "hit_rate": 0.5, "size": 2}
    ),
])
def test_cache_stats_scenarios(sample_datapoint, scenario, actions, expected_stats):
    """Test various cache statistics scenarios."""
    cache = DatasetCache()
    
    # Perform actions
    for action, key in actions:
        if action == "put":
            cache.put(key, sample_datapoint)
        elif action == "get":
            cache.get(key)
    
    # Verify stats
    stats = cache.get_stats()
    for key, expected_value in expected_stats.items():
        if isinstance(expected_value, float):
            assert abs(stats[key] - expected_value) < 1.0e-05, f"{key} mismatch"
        else:
            assert stats[key] == expected_value, f"{key} mismatch"
