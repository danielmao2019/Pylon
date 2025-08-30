import pickle
import torch
from data.cache.cpu_dataset_cache import CPUDatasetCache


def test_cpu_cache_pickle_basic():
    """Test that CPUDatasetCache can be pickled and unpickled without thread lock issues."""
    cache = CPUDatasetCache(max_memory_percent=50.0, enable_validation=True)

    # Test pickling without any data
    pickled_data = pickle.dumps(cache)
    unpickled_cache = pickle.loads(pickled_data)

    assert unpickled_cache.max_memory_percent == 50.0
    assert unpickled_cache.enable_validation is True
    assert len(unpickled_cache.cache) == 0
    assert unpickled_cache.total_memory == 0


def test_cpu_cache_pickle_with_data():
    """Test that CPUDatasetCache can be pickled and then used to store data after unpickling."""
    cache1 = CPUDatasetCache(max_memory_percent=50.0, enable_validation=True)

    # Add some test data to first cache
    test_data = {
        'inputs': {'tensor': torch.randn(10, 3)},
        'labels': {'label': torch.tensor([1])},
        'meta_info': {'idx': 0}
    }
    cache1.put(0, test_data)
    assert cache1.get_size() == 1

    # Create a second cache and pickle it
    cache2 = CPUDatasetCache(max_memory_percent=50.0, enable_validation=True)

    # Test pickling fresh cache
    pickled_data = pickle.dumps(cache2)
    unpickled_cache = pickle.loads(pickled_data)

    # Now add data to unpickled cache
    unpickled_cache.put(0, test_data)
    assert unpickled_cache.get_size() == 1

    # Verify cached data works correctly
    retrieved_data = unpickled_cache.get(0)
    assert retrieved_data is not None
    assert torch.equal(retrieved_data['inputs']['tensor'], test_data['inputs']['tensor'])


def test_cpu_cache_pickle_lock_recreation():
    """Test that thread lock is properly recreated after unpickling."""
    cache = CPUDatasetCache(max_memory_percent=50.0, enable_validation=False)

    # Verify lock is initialized immediately via BaseCache
    original_lock = cache._lock
    assert original_lock is not None

    # Pickle and unpickle
    pickled_data = pickle.dumps(cache)
    unpickled_cache = pickle.loads(pickled_data)

    # Verify lock is recreated after unpickling (different object)
    new_lock = unpickled_cache._lock
    assert new_lock is not None
    assert new_lock is not original_lock  # Should be a different lock object

    # Verify lock works for thread-safe operations
    test_data = {
        'inputs': {'tensor': torch.randn(2, 2)},
        'labels': {'label': torch.tensor([1])},
        'meta_info': {'idx': 0}
    }
    unpickled_cache.put(0, test_data)
    retrieved = unpickled_cache.get(0)
    assert retrieved is not None


def test_cpu_cache_pickle_logger_preservation():
    """Test that logger is properly preserved after unpickling."""
    cache = CPUDatasetCache(max_memory_percent=50.0, enable_validation=False)

    # Verify logger is initialized immediately via BaseCache
    original_logger = cache.logger
    assert original_logger is not None

    # Pickle and unpickle
    pickled_data = pickle.dumps(cache)
    unpickled_cache = pickle.loads(pickled_data)

    # Verify logger is preserved after unpickling (loggers are singletons)
    preserved_logger = unpickled_cache.logger
    assert preserved_logger is not None
    assert preserved_logger is original_logger  # Should be the same logger object (singleton)

    # Verify logger works for logging operations
    test_data = {
        'inputs': {'tensor': torch.randn(2, 2)},
        'labels': {'label': torch.tensor([1])},
        'meta_info': {'idx': 0}
    }
    unpickled_cache.put(0, test_data)
    # Logger should work without issues during cache operations


def test_cpu_cache_pickle_thread_safety_after_unpickling():
    """Test that cache operations work correctly after unpickling (thread safety)."""
    cache = CPUDatasetCache(max_memory_percent=50.0, enable_validation=True)

    # Pickle cache
    pickled_data = pickle.dumps(cache)
    unpickled_cache = pickle.loads(pickled_data)

    # Test cache operations after unpickling
    test_data1 = {
        'inputs': {'tensor': torch.randn(5, 2)},
        'labels': {'label': torch.tensor([0])},
        'meta_info': {'idx': 1}
    }
    test_data2 = {
        'inputs': {'tensor': torch.randn(3, 4)},
        'labels': {'label': torch.tensor([2])},
        'meta_info': {'idx': 2}
    }

    # These operations should work without issues with the recreated lock
    unpickled_cache.put(1, test_data1)
    unpickled_cache.put(2, test_data2)
    retrieved_data1 = unpickled_cache.get(1)
    retrieved_data2 = unpickled_cache.get(2)

    assert retrieved_data1 is not None
    assert retrieved_data2 is not None
    assert unpickled_cache.get_size() == 2

    # Verify data integrity
    assert torch.equal(retrieved_data1['inputs']['tensor'], test_data1['inputs']['tensor'])
    assert torch.equal(retrieved_data2['inputs']['tensor'], test_data2['inputs']['tensor'])


def test_cpu_cache_pickle_multiprocessing_compatibility():
    """Test that CPUDatasetCache works correctly with multiprocessing after unpickling."""
    # Test multiprocessing compatibility

    def worker_function(cache):
        """Worker function that operates on the cache."""
        # Add data from worker process
        test_data = {
            'inputs': {'tensor': torch.randn(2, 2)},
            'labels': {'label': torch.tensor([3])},
            'meta_info': {'idx': 3}
        }
        cache.put(3, test_data)
        return cache.get_size()

    cache = CPUDatasetCache(max_memory_percent=50.0, enable_validation=False)

    # Pickle and unpickle to simulate multiprocessing serialization
    pickled_data = pickle.dumps(cache)
    unpickled_cache = pickle.loads(pickled_data)

    # Test that cache operations work in a separate process context
    # Note: This tests the pickle compatibility; actual multiprocessing would require more setup
    result = worker_function(unpickled_cache)
    assert result == 1

    # Verify the cache has the expected data
    retrieved_data = unpickled_cache.get(3)
    assert retrieved_data is not None
    assert retrieved_data['labels']['label'].item() == 3
