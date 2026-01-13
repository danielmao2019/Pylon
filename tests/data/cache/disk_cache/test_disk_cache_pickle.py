import os
import pickle
import tempfile

import torch

from data.cache.disk_dataset_cache import DiskDatasetCache
from utils.io.json import load_json


def _fp(cache: DiskDatasetCache, idx: int) -> str:
    return os.path.join(cache.version_dir, f"{idx}.pt")


def test_disk_cache_pickle_basic():
    """Test that DiskDatasetCache can be pickled and unpickled without thread lock issues."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create cache instance
        cache = DiskDatasetCache(
            cache_dir=temp_dir,
            version_hash="test_version_123",
            enable_validation=True,
            dataset_class_name="TestDataset",
            version_dict={"param1": "value1"},
        )

        # Verify lock was initialized during construction via BaseCache
        original_lock = cache._lock
        assert original_lock is not None

        # Test pickling
        pickled_data = pickle.dumps(cache)
        unpickled_cache = pickle.loads(pickled_data)

        assert unpickled_cache.cache_dir == temp_dir
        assert unpickled_cache.version_hash == "test_version_123"
        assert unpickled_cache.enable_validation is True
        assert unpickled_cache.dataset_class_name == "TestDataset"
        assert unpickled_cache.version_dict == {"param1": "value1"}
        assert unpickled_cache.get_size() == 0

        # Verify lock was recreated after unpickling (different object)
        new_lock = unpickled_cache._lock
        assert new_lock is not None
        assert new_lock is not original_lock  # Should be a different lock object


def test_disk_cache_pickle_with_data():
    """Test that DiskDatasetCache can be pickled and then used to retrieve data after unpickling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DiskDatasetCache(
            cache_dir=temp_dir, version_hash="test_version_456", enable_validation=True
        )

        # Add some test data
        test_data = {
            'inputs': {'tensor': torch.randn(10, 3)},
            'labels': {'label': torch.tensor([1])},
            'meta_info': {'idx': 0},
        }
        cache.put(test_data, cache_filepath=_fp(cache, 0))

        # Verify data exists on disk
        assert cache.exists(_fp(cache, 0))
        assert cache.get_size() == 1

        # Test pickling with data
        pickled_data = pickle.dumps(cache)
        unpickled_cache = pickle.loads(pickled_data)

        assert unpickled_cache.get_size() == 1
        assert unpickled_cache.exists(_fp(unpickled_cache, 0))

        # Verify cached data is preserved
        retrieved_data = unpickled_cache.get(cache_filepath=_fp(unpickled_cache, 0))
        assert retrieved_data is not None
        assert torch.equal(
            retrieved_data['inputs']['tensor'], test_data['inputs']['tensor']
        )


def test_disk_cache_pickle_lock_recreation():
    """Test that thread lock is properly recreated after unpickling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DiskDatasetCache(
            cache_dir=temp_dir, version_hash="test_version_789", enable_validation=False
        )

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
            'meta_info': {'idx': 0},
        }
        unpickled_cache.put(test_data, cache_filepath=_fp(unpickled_cache, 0))
        retrieved = unpickled_cache.get(cache_filepath=_fp(unpickled_cache, 0))
        assert retrieved is not None


def test_disk_cache_pickle_thread_safety_after_unpickling():
    """Test that cache operations work correctly after unpickling (thread safety)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DiskDatasetCache(
            cache_dir=temp_dir, version_hash="test_version_abc", enable_validation=True
        )

        # Add test data before pickling
        test_data1 = {
            'inputs': {'tensor': torch.randn(5, 2)},
            'labels': {'label': torch.tensor([0])},
            'meta_info': {'idx': 1},
        }
        cache.put(test_data1, cache_filepath=_fp(cache, 1))

        # Pickle and unpickle
        pickled_data = pickle.dumps(cache)
        unpickled_cache = pickle.loads(pickled_data)

        # Test cache operations after unpickling
        test_data2 = {
            'inputs': {'tensor': torch.randn(3, 4)},
            'labels': {'label': torch.tensor([2])},
            'meta_info': {'idx': 2},
        }

        # These operations should work without issues with the recreated lock
        unpickled_cache.put(test_data2, cache_filepath=_fp(unpickled_cache, 2))
        retrieved_data1 = unpickled_cache.get(cache_filepath=_fp(unpickled_cache, 1))
        retrieved_data2 = unpickled_cache.get(cache_filepath=_fp(unpickled_cache, 2))

        assert retrieved_data1 is not None
        assert retrieved_data2 is not None
        assert unpickled_cache.get_size() == 2

        # Verify data integrity
        assert torch.equal(
            retrieved_data1['inputs']['tensor'], test_data1['inputs']['tensor']
        )
        assert torch.equal(
            retrieved_data2['inputs']['tensor'], test_data2['inputs']['tensor']
        )


def test_disk_cache_pickle_metadata_preservation():
    """Test that metadata is properly preserved after pickling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DiskDatasetCache(
            cache_dir=temp_dir,
            version_hash="test_version_def",
            enable_validation=True,
            dataset_class_name="TestDataset",
            version_dict={"param1": "value1", "param2": 42},
        )

        # Let metadata be written (happens in constructor)
        original_metadata = load_json(cache.metadata_file)

        # Pickle and unpickle
        pickled_data = pickle.dumps(cache)
        unpickled_cache = pickle.loads(pickled_data)

        # Verify metadata is preserved
        preserved_metadata = load_json(unpickled_cache.metadata_file)
        assert preserved_metadata == original_metadata

        # Verify specific fields
        version_info = preserved_metadata["test_version_def"]
        assert version_info["dataset_class_name"] == "TestDataset"
        assert version_info["version_dict"] == {"param1": "value1", "param2": 42}
        assert version_info["enable_validation"] is True


def test_disk_cache_pickle_multiprocessing_compatibility():
    """Test that DiskDatasetCache works correctly with multiprocessing after unpickling."""

    def worker_function(cache):
        """Worker function that operates on the cache."""
        # Add data from worker process
        test_data = {
            'inputs': {'tensor': torch.randn(2, 2)},
            'labels': {'label': torch.tensor([3])},
            'meta_info': {'idx': 3},
        }
        cache.put(test_data, cache_filepath=_fp(cache, 3))
        return cache.get_size()

    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DiskDatasetCache(
            cache_dir=temp_dir, version_hash="test_version_ghi", enable_validation=False
        )

        # Pickle and unpickle to simulate multiprocessing serialization
        pickled_data = pickle.dumps(cache)
        unpickled_cache = pickle.loads(pickled_data)

        # Test that cache operations work in a separate process context
        # Note: This tests the pickle compatibility; actual multiprocessing would require more setup
        result = worker_function(unpickled_cache)
        assert result == 1

        # Verify the cache has the expected data
        retrieved_data = unpickled_cache.get(cache_filepath=_fp(unpickled_cache, 3))
        assert retrieved_data is not None
        assert retrieved_data['labels']['label'].item() == 3


def test_disk_cache_pickle_validated_keys_reset():
    """Test that validated_keys set is properly handled during pickling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DiskDatasetCache(
            cache_dir=temp_dir, version_hash="test_version_jkl", enable_validation=True
        )

        # Add data and trigger validation
        test_data = {
            'inputs': {'tensor': torch.randn(3, 3)},
            'labels': {'label': torch.tensor([5])},
            'meta_info': {'idx': 5},
        }
        cache.put(test_data, cache_filepath=_fp(cache, 5))

        # Get data to trigger validation and populate validated_keys
        retrieved = cache.get(cache_filepath=_fp(cache, 5))
        assert retrieved is not None
        assert _fp(cache, 5) in cache.validated_keys

        # Pickle and unpickle
        pickled_data = pickle.dumps(cache)
        unpickled_cache = pickle.loads(pickled_data)

        # Verify validated_keys is preserved
        assert _fp(unpickled_cache, 5) in unpickled_cache.validated_keys

        # Verify subsequent access works (should skip validation)
        retrieved_again = unpickled_cache.get(cache_filepath=_fp(unpickled_cache, 5))
        assert retrieved_again is not None
        assert torch.equal(
            retrieved_again['inputs']['tensor'], test_data['inputs']['tensor']
        )


def test_disk_cache_pickle_logger_preservation():
    """Test that logger is properly preserved after unpickling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DiskDatasetCache(
            cache_dir=temp_dir,
            version_hash="test_version_logger",
            enable_validation=False,
        )

        # Verify logger is initialized immediately via BaseCache
        original_logger = cache.logger
        assert original_logger is not None

        # Pickle and unpickle
        pickled_data = pickle.dumps(cache)
        unpickled_cache = pickle.loads(pickled_data)

        # Verify logger is preserved after unpickling (loggers are singletons)
        preserved_logger = unpickled_cache.logger
        assert preserved_logger is not None
        assert (
            preserved_logger is original_logger
        )  # Should be the same logger object (singleton)

        # Verify logger works for logging operations
        test_data = {
            'inputs': {'tensor': torch.randn(2, 2)},
            'labels': {'label': torch.tensor([1])},
            'meta_info': {'idx': 0},
        }
        unpickled_cache.put(test_data, cache_filepath=_fp(unpickled_cache, 0))
        # Logger should work without issues during cache operations
