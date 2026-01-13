"""Test disk cache validation and corruption detection."""

import pytest
import tempfile
import os
import torch
from data.cache.disk_dataset_cache import DiskDatasetCache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_datapoint():
    """Create a sample datapoint for testing."""
    return {
        'inputs': {
            'image': torch.randn(3, 64, 64),
            'features': torch.randn(256)
        },
        'labels': {
            'mask': torch.randint(0, 2, (64, 64))
        },
        'meta_info': {
            'path': '/test/image.jpg',
            'index': 42,
            'dataset': 'test'
        }
    }


def test_checksum_validation_enabled(temp_cache_dir, sample_datapoint):
    """Test checksum validation when enabled."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="validation_test",
        enable_validation=True
    )

    cache_file = os.path.join(cache.version_dir, "0.pt")
    cache.put(sample_datapoint, cache_filepath=cache_file)

    # Retrieve should succeed with valid checksum
    result = cache.get(cache_filepath=cache_file)
    assert result is not None

    # Load cached file and verify checksum is stored
    cached_data = torch.load(cache_file)
    assert 'checksum' in cached_data
    assert len(cached_data['checksum']) > 0


def test_checksum_validation_disabled(temp_cache_dir, sample_datapoint):
    """Test behavior when validation is disabled."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="no_validation_test",
        enable_validation=False
    )

    cache_file = os.path.join(cache.version_dir, "0.pt")
    cache.put(sample_datapoint, cache_filepath=cache_file)

    # Load cached file and verify no checksum
    cached_data = torch.load(cache_file)
    assert 'checksum' not in cached_data

    # Retrieve should work without validation
    result = cache.get(cache_filepath=cache_file)
    assert result is not None


def test_corruption_detection_and_removal(temp_cache_dir, sample_datapoint):
    """Test detection and handling of corrupted cache files - fails fast and loud."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="corruption_test",
        enable_validation=True
    )

    cache.put(sample_datapoint, cache_filepath=os.path.join(cache.version_dir, "0.pt"))
    cache_file = os.path.join(cache.version_dir, "0.pt")
    assert os.path.exists(cache_file)

    # Corrupt the file by writing invalid data
    with open(cache_file, 'w') as f:
        f.write("This is not valid torch data")

    # Attempt to retrieve should raise RuntimeError - fail fast and loud
    with pytest.raises(RuntimeError, match="Error loading torch file"):
        cache.get(cache_filepath=cache_file)

    # File should still exist - we don't mask errors by removing files
    assert os.path.exists(cache_file)


def test_validation_session_tracking(temp_cache_dir, sample_datapoint):
    """Test that validation only happens once per session."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="session_test",
        enable_validation=True
    )

    cache_file = os.path.join(cache.version_dir, "0.pt")
    cache.put(sample_datapoint, cache_filepath=cache_file)

    # First access should validate
    assert cache_file not in cache.validated_keys
    result1 = cache.get(cache_filepath=cache_file)
    assert result1 is not None
    assert cache_file in cache.validated_keys

    # Mock the validation method to track calls
    from unittest.mock import patch
    with patch.object(cache, '_validate_item', wraps=cache._validate_item) as mock_validate:
        # Second access should skip validation
        result2 = cache.get(cache_filepath=cache_file)
        assert result2 is not None
        mock_validate.assert_called_once()  # Should be called but return early


def test_checksum_computation_consistency(temp_cache_dir, sample_datapoint):
    """Test that checksum computation is consistent."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="checksum_test",
        enable_validation=True
    )

    # Compute checksum multiple times for same data
    checksum1 = cache._compute_checksum(sample_datapoint)
    checksum2 = cache._compute_checksum(sample_datapoint)
    checksum3 = cache._compute_checksum(sample_datapoint)

    # All checksums should be identical
    assert checksum1 == checksum2 == checksum3
    assert len(checksum1) > 0


def test_different_data_different_checksums(temp_cache_dir, sample_datapoint):
    """Test that different data produces different checksums."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="diff_checksum_test",
        enable_validation=True
    )

    # Create modified datapoint
    modified_datapoint = {
        'inputs': {
            'image': sample_datapoint['inputs']['image'] + 0.1,  # Slight modification
            'features': sample_datapoint['inputs']['features']
        },
        'labels': sample_datapoint['labels'],
        'meta_info': sample_datapoint['meta_info']
    }

    # Compute checksums
    checksum1 = cache._compute_checksum(sample_datapoint)
    checksum2 = cache._compute_checksum(modified_datapoint)

    # Checksums should be different
    assert checksum1 != checksum2


def test_validation_with_corrupted_checksum(temp_cache_dir, sample_datapoint):
    """Test validation failure when stored checksum is corrupted - fails fast and loud."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="corrupted_checksum_test",
        enable_validation=True
    )

    cache_file = os.path.join(cache.version_dir, "0.pt")
    cache.put(sample_datapoint, cache_filepath=cache_file)

    # Load and modify the stored checksum
    cached_data = torch.load(cache_file)
    cached_data['checksum'] = "invalid_checksum"
    torch.save(cached_data, cache_file)

    # Validation should fail fast and loud - ValueError is raised immediately
    with pytest.raises(ValueError, match="validation failed"):
        cache.get(cache_filepath=cache_file)

    # File should still exist - we don't mask errors by removing files
    assert os.path.exists(cache_file)


def test_validation_reset_on_cache_restart(temp_cache_dir, sample_datapoint):
    """Test that validation tracking resets when cache is recreated."""
    version_hash = "restart_test"

    # Create first cache instance
    cache1 = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash=version_hash,
        enable_validation=True
    )

    cache_file = os.path.join(cache1.version_dir, "0.pt")
    cache1.put(sample_datapoint, cache_filepath=cache_file)
    result1 = cache1.get(cache_filepath=cache_file)
    assert result1 is not None
    assert cache_file in cache1.validated_keys

    # Create new cache instance (simulating restart)
    cache2 = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash=version_hash,
        enable_validation=True
    )

    # Validation tracking should be reset
    assert len(cache2.validated_keys) == 0

    # Item should still be retrievable and validation should work
    result2 = cache2.get(cache_filepath=cache_file)
    assert result2 is not None
    assert cache_file in cache2.validated_keys
