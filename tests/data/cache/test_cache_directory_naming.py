"""Test cache directory naming for datasets with and without data_root."""

import os
import json
import tempfile
import pytest


def test_cache_directory_with_data_root(temp_data_root, TestDatasetWithDataRoot):
    """Test cache directory naming when data_root is provided."""
    dataset = TestDatasetWithDataRoot(
        data_root=temp_data_root,
        split='train',
        use_disk_cache=True,
        use_cpu_cache=False
    )
    
    # Access a datapoint to trigger cache creation
    _ = dataset[0]
    
    # Check cache directory naming
    assert dataset.cache is not None
    assert dataset.cache.disk_cache is not None
    
    cache_info = dataset.cache.get_info()
    expected_cache_dir = f"{temp_data_root}_cache"
    
    assert cache_info['cache_dir'] == expected_cache_dir, \
        f"Expected cache dir: {expected_cache_dir}, got: {cache_info['cache_dir']}"
    
    # Verify the cache directory exists
    assert os.path.exists(expected_cache_dir), \
        f"Cache directory should exist: {expected_cache_dir}"


def test_cache_directory_without_data_root(TestDatasetWithoutDataRoot):
    """Test cache directory naming when data_root is NOT provided."""
    dataset = TestDatasetWithoutDataRoot(
        split='train',
        use_disk_cache=True,
        use_cpu_cache=False
    )
    
    # Access a datapoint to trigger cache creation
    _ = dataset[0]
    
    # Check cache directory naming
    assert dataset.cache is not None
    assert dataset.cache.disk_cache is not None
    
    cache_info = dataset.cache.get_info()
    expected_cache_dir = f"/tmp/cache/{dataset.__class__.__name__.lower()}_cache"
    
    assert cache_info['cache_dir'] == expected_cache_dir, \
        f"Expected cache dir: {expected_cache_dir}, got: {cache_info['cache_dir']}"
    
    # Verify the cache directory exists
    assert os.path.exists(expected_cache_dir), \
        f"Cache directory should exist: {expected_cache_dir}"


def test_cache_directory_with_soft_link(temp_data_root, TestDatasetWithDataRoot):
    """Test cache directory naming when data_root is a soft link."""
    # Create a soft link to the temp data root
    with tempfile.TemporaryDirectory() as link_parent:
        link_path = os.path.join(link_parent, "data_link")
        os.symlink(temp_data_root, link_path)
        
        dataset = TestDatasetWithDataRoot(
            data_root=link_path,
            split='train',
            use_disk_cache=True,
            use_cpu_cache=False
        )
        
        # Access a datapoint to trigger cache creation
        _ = dataset[0]
        
        # Check cache directory naming - should resolve the soft link
        assert dataset.cache is not None
        assert dataset.cache.disk_cache is not None
        
        cache_info = dataset.cache.get_info()
        expected_cache_dir = f"{temp_data_root}_cache"  # Resolved path
        
        assert cache_info['cache_dir'] == expected_cache_dir, \
            f"Expected resolved cache dir: {expected_cache_dir}, got: {cache_info['cache_dir']}"


def test_cache_metadata_includes_dataset_info(TestDatasetWithoutDataRoot):
    """Test that cache metadata includes dataset class name and version dict."""
    dataset = TestDatasetWithoutDataRoot(
        split='train',
        base_seed=42,
        use_disk_cache=True,
        use_cpu_cache=False
    )
    
    # Access a datapoint to trigger cache creation
    _ = dataset[0]
    
    # Check metadata file content
    assert dataset.cache is not None
    assert dataset.cache.disk_cache is not None
    
    metadata_file = dataset.cache.disk_cache.metadata_file
    
    assert os.path.exists(metadata_file), \
        f"Metadata file should exist: {metadata_file}"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    version_hash = dataset.get_cache_version_hash()
    assert version_hash in metadata, \
        f"Version hash {version_hash} should be in metadata"
    
    version_info = metadata[version_hash]
    
    # Check required fields
    assert 'dataset_class_name' in version_info, \
        "Metadata should include dataset_class_name"
    assert version_info['dataset_class_name'] == 'TestDatasetWithoutDataRootImpl', \
        f"Expected class name: TestDatasetWithoutDataRootImpl, got: {version_info['dataset_class_name']}"
    
    assert 'version_dict' in version_info, \
        "Metadata should include version_dict"
    
    version_dict = version_info['version_dict']
    assert 'class_name' in version_dict, \
        "Version dict should include class_name"
    assert 'split' in version_dict, \
        "Version dict should include split"  
    assert 'base_seed' in version_dict, \
        "Version dict should include base_seed"
    assert version_dict['base_seed'] == 42, \
        f"Expected base_seed: 42, got: {version_dict['base_seed']}"


def test_multiple_datasets_different_cache_dirs(temp_data_root, temp_dir_for_dataset1, TestDatasetWithoutDataRoot, TestDatasetWithDataRoot):
    """Test that different dataset classes get different cache directories."""
    # Create dataset1 with temp directory as data_root (to avoid /tmp/cache/ race conditions)
    dataset1 = TestDatasetWithoutDataRoot(
        data_root=temp_dir_for_dataset1,
        split='train',
        use_disk_cache=True,
        use_cpu_cache=False
    )
    
    dataset2 = TestDatasetWithDataRoot(
        data_root=temp_data_root,
        split='train',
        use_disk_cache=True,
        use_cpu_cache=False
    )
    
    # Get cache directory names
    cache_info1 = dataset1.cache.get_info()
    cache_info2 = dataset2.cache.get_info()
    
    # They should have different cache directories
    assert cache_info1['cache_dir'] != cache_info2['cache_dir'], \
        f"Different datasets should have different cache dirs: {cache_info1['cache_dir']} vs {cache_info2['cache_dir']}"
    
    # Dataset1 should use the overridden temp directory
    expected_dir1 = f"{temp_dir_for_dataset1}_cache"
    assert cache_info1['cache_dir'] == expected_dir1, \
        f"Expected: {expected_dir1}, got: {cache_info1['cache_dir']}"
    
    # Dataset2 with data_root should use data_root pattern
    expected_dir2 = f"{temp_data_root}_cache"
    assert cache_info2['cache_dir'] == expected_dir2, \
        f"Expected: {expected_dir2}, got: {cache_info2['cache_dir']}"


@pytest.mark.parametrize("split,base_seed", [
    ('train', 0),
    ('train', 42),
    ('test', 0),
])
def test_cache_directory_naming_consistency(split, base_seed, TestDatasetWithoutDataRoot):
    """Test that cache directory naming is consistent across multiple instantiations."""
    # Create dataset twice with same config
    dataset1 = TestDatasetWithoutDataRoot(
        split=split,
        base_seed=base_seed,
        use_disk_cache=True,
        use_cpu_cache=False
    )
    dataset2 = TestDatasetWithoutDataRoot(
        split=split,
        base_seed=base_seed,
        use_disk_cache=True,
        use_cpu_cache=False
    )
    
    # They should have the same cache directory
    cache_info1 = dataset1.cache.get_info()
    cache_info2 = dataset2.cache.get_info()
    
    assert cache_info1['cache_dir'] == cache_info2['cache_dir'], \
        f"Same config should result in same cache dir: {cache_info1['cache_dir']} vs {cache_info2['cache_dir']}"
    
    # And same version hashes
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    assert hash1 == hash2, \
        f"Same config should result in same version hash: {hash1} vs {hash2}"


@pytest.fixture(autouse=True)
def cleanup_test_cache_dirs():
    """Clean up test cache directories after each test."""
    yield
    
    # Clean up any test cache directories that might have been created
    test_cache_dirs = [
        '/tmp/cache/testdatasetwithdatarootimpl_cache',
        '/tmp/cache/testdatasetwithoutdatarootimpl_cache'
    ]
    
    for cache_dir in test_cache_dirs:
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def temp_data_root():
    """Create a temporary directory to use as data root."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def temp_dir_for_dataset1():
    """Create a temporary directory for dataset1 testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    cache_dir = f"{temp_dir}_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
