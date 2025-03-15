import pytest
import torch
import threading
from data.cache import DatasetCache


def test_cache_thread_safety(sample_datapoint):
    """Test thread safety of cache operations."""
    cache = DatasetCache()
    num_threads = 10
    ops_per_thread = 100
    errors = []
    
    def worker(thread_id):
        try:
            for i in range(ops_per_thread):
                key = (thread_id * ops_per_thread + i) % 5
                if i % 2 == 0:
                    cache.put(key, sample_datapoint)
                else:
                    result = cache.get(key)
                    if result is not None:
                        # Verify data integrity
                        assert torch.all(result['inputs']['image'] == sample_datapoint['inputs']['image'])
        except Exception as e:
            errors.append(f"Thread {thread_id} error: {str(e)}")
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert not errors, f"Thread errors occurred: {errors}"
    
    # Verify cache is in valid state
    assert len(cache.cache) <= 5
    stats = cache.get_stats()
    assert stats['hits'] + stats['misses'] == (num_threads * ops_per_thread) // 2
