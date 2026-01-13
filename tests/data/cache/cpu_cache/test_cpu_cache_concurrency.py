import torch
import threading
from data.cache.cpu_dataset_cache import CPUDatasetCache


def test_cache_thread_safety(sample_datapoint, cache_key_factory):
    """Test thread safety of cache operations."""
    cache = CPUDatasetCache()
    num_threads = 3
    ops_per_thread = 6
    errors = []

    def worker(thread_id):
        try:
            for i in range(ops_per_thread):
                key = i % 2  # Two keys for better coverage
                cache_key = cache_key_factory(key)
                if i % 2 == 0:
                    cache.put(sample_datapoint, cache_filepath=cache_key)
                else:
                    result = cache.get(cache_filepath=cache_key)
                    if result is not None:
                        assert torch.all(result['inputs']['image'] == sample_datapoint['inputs']['image'])
        except Exception as e:
            errors.append(f"Thread {thread_id} error: {str(e)}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors occurred: {errors}"

    # Verify final cache state
    assert len(cache.cache) <= 2
