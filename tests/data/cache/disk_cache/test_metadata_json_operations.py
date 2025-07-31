"""Test disk cache metadata JSON file operations."""

import pytest
import tempfile
import os
from data.cache.disk_dataset_cache import DiskDatasetCache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_datapoint():
    """Create a sample datapoint for testing."""
    import torch
    return {
        'inputs': {'image': torch.randn(3, 64, 64)},
        'labels': {'mask': torch.randint(0, 2, (64, 64))},
        'meta_info': {'path': '/test/image.jpg', 'index': 42}
    }


def test_metadata_file_creation(temp_cache_dir):
    """Test that metadata JSON file is created properly."""
    version_hash = "test_version_123"
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash=version_hash,
        enable_validation=True
    )
    
    # Verify metadata file exists
    metadata_file = os.path.join(temp_cache_dir, 'cache_metadata.json')
    assert os.path.exists(metadata_file)
    assert cache.metadata_file == metadata_file


def test_metadata_json_content(temp_cache_dir):
    """Test metadata JSON file contains correct information."""
    version_hash = "content_test_456"
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash=version_hash,
        enable_validation=False
    )
    
    # Check metadata content
    metadata = cache.get_metadata()
    assert version_hash in metadata
    assert 'created_at' in metadata[version_hash]
    assert 'cache_dir' in metadata[version_hash]
    assert 'version_dir' in metadata[version_hash]
    assert 'enable_validation' in metadata[version_hash]
    assert metadata[version_hash]['enable_validation'] is False


def test_multiple_versions_in_metadata(temp_cache_dir, sample_datapoint):
    """Test that multiple cache versions are tracked in the same metadata file."""
    versions = ["version_1", "version_2", "version_3"]
    caches = []
    
    # Create multiple cache versions
    for i, version in enumerate(versions):
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash=version,
            enable_validation=(i == 1)  # Mix validation settings
        )
        caches.append(cache)
        cache.put(0, sample_datapoint)
    
    # Check that all versions are tracked in metadata
    metadata = caches[-1].get_metadata()
    for version in versions:
        assert version in metadata
        assert 'created_at' in metadata[version]
        assert 'cache_dir' in metadata[version]
        assert 'version_dir' in metadata[version]
        assert 'enable_validation' in metadata[version]
    
    # Verify validation settings are tracked correctly
    assert metadata["version_1"]['enable_validation'] is False
    assert metadata["version_2"]['enable_validation'] is True
    assert metadata["version_3"]['enable_validation'] is False


def test_metadata_persistence_across_instances(temp_cache_dir, sample_datapoint):
    """Test that metadata persists across different cache instances."""
    version_hash = "persistence_test"
    
    # Create first cache instance and add some data
    cache1 = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash=version_hash,
        enable_validation=True
    )
    cache1.put(0, sample_datapoint)
    
    # Get metadata from first instance
    metadata1 = cache1.get_metadata()
    assert version_hash in metadata1
    creation_time1 = metadata1[version_hash]['created_at']
    
    # Create second cache instance with same version
    cache2 = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash=version_hash,
        enable_validation=True
    )
    
    # Metadata should be the same
    metadata2 = cache2.get_metadata()
    assert version_hash in metadata2
    assert metadata2[version_hash]['created_at'] == creation_time1
    
    # Data should be retrievable from second instance
    result = cache2.get(0)
    assert result is not None


def test_metadata_corruption_handling(temp_cache_dir):
    """Test handling of corrupted metadata files - fails fast and loud."""
    # Create cache to generate initial metadata
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="corruption_test",
        enable_validation=False
    )
    
    # Corrupt the metadata file
    metadata_file = cache.metadata_file
    with open(metadata_file, 'w') as f:
        f.write("invalid json content {")
    
    # Trying to load corrupted metadata should fail fast and loud
    with pytest.raises(RuntimeError, match="Error loading JSON"):
        cache.get_metadata()
    
    # Creating new cache should also fail when trying to load corrupted metadata
    with pytest.raises(RuntimeError, match="Error loading JSON"):
        DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="corruption_test_2",
            enable_validation=False
        )


def test_metadata_file_permissions(temp_cache_dir):
    """Test metadata file creation with proper permissions."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="permissions_test",
        enable_validation=False
    )
    
    # Check that metadata file exists and is readable
    metadata_file = cache.metadata_file
    assert os.path.exists(metadata_file)
    assert os.access(metadata_file, os.R_OK)
    assert os.access(metadata_file, os.W_OK)
    
    # Verify file has reasonable permissions
    stat_info = os.stat(metadata_file)
    # Should be readable/writable by owner at minimum
    assert stat_info.st_mode & 0o600 == 0o600