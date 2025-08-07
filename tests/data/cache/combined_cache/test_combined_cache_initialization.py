"""Test combined cache initialization and configuration."""

import pytest
import os
from data.cache.combined_dataset_cache import CombinedDatasetCache
from data.cache.cpu_dataset_cache import CPUDatasetCache
from data.cache.disk_dataset_cache import DiskDatasetCache


def test_cache_initialization_both_enabled(temp_cache_setup):
    """Test cache initialization with both CPU and disk caches enabled."""
    cache = CombinedDatasetCache(
        data_root=temp_cache_setup['data_root'],
        version_hash=temp_cache_setup['version_hash'],
        use_cpu_cache=True,
        use_disk_cache=True
    )
    
    # Verify configuration
    assert cache.use_cpu_cache is True
    assert cache.use_disk_cache is True
    
    # Verify component initialization
    assert cache.cpu_cache is not None
    assert cache.disk_cache is not None
    assert isinstance(cache.cpu_cache, CPUDatasetCache)
    assert isinstance(cache.disk_cache, DiskDatasetCache)


def test_cache_initialization_cpu_only(temp_cache_setup):
    """Test cache initialization with only CPU cache enabled."""
    cache = CombinedDatasetCache(
        data_root=temp_cache_setup['data_root'],
        version_hash=temp_cache_setup['version_hash'],
        use_cpu_cache=True,
        use_disk_cache=False
    )
    
    # Verify configuration
    assert cache.use_cpu_cache is True
    assert cache.use_disk_cache is False
    
    # Verify component initialization
    assert cache.cpu_cache is not None
    assert cache.disk_cache is None
    assert isinstance(cache.cpu_cache, CPUDatasetCache)


def test_cache_initialization_disk_only(temp_cache_setup):
    """Test cache initialization with only disk cache enabled."""
    cache = CombinedDatasetCache(
        data_root=temp_cache_setup['data_root'],
        version_hash=temp_cache_setup['version_hash'],
        use_cpu_cache=False,
        use_disk_cache=True
    )
    
    # Verify configuration
    assert cache.use_cpu_cache is False
    assert cache.use_disk_cache is True
    
    # Verify component initialization
    assert cache.cpu_cache is None
    assert cache.disk_cache is not None
    assert isinstance(cache.disk_cache, DiskDatasetCache)


def test_cache_initialization_both_disabled(temp_cache_setup):
    """Test cache initialization with both caches disabled."""
    cache = CombinedDatasetCache(
        data_root=temp_cache_setup['data_root'],
        version_hash=temp_cache_setup['version_hash'],
        use_cpu_cache=False,
        use_disk_cache=False
    )
    
    # Verify configuration
    assert cache.use_cpu_cache is False
    assert cache.use_disk_cache is False
    
    # Verify component initialization
    assert cache.cpu_cache is None
    assert cache.disk_cache is None


def test_cache_initialization_all_configurations(all_cache_configurations, cache_config_factory):
    """Test all combinations of cache enable/disable configurations."""
    for use_cpu, use_disk, description in all_cache_configurations:
        cache = cache_config_factory(
            use_cpu_cache=use_cpu,
            use_disk_cache=use_disk
        )
        
        # Verify configuration matches expected
        assert cache.use_cpu_cache == use_cpu, f"CPU cache config mismatch for {description}"
        assert cache.use_disk_cache == use_disk, f"Disk cache config mismatch for {description}"
        
        # Verify component initialization
        assert (cache.cpu_cache is not None) == use_cpu, f"CPU cache component mismatch for {description}"
        assert (cache.disk_cache is not None) == use_disk, f"Disk cache component mismatch for {description}"


def test_cache_initialization_parameter_propagation(temp_cache_setup):
    """Test that parameters are correctly propagated to underlying caches."""
    max_memory_percent = 50.0
    enable_cpu_validation = True
    enable_disk_validation = True
    dataset_class_name = 'CustomDataset'
    version_dict = {'param1': 'value1', 'param2': 42}
    
    cache = CombinedDatasetCache(
        data_root=temp_cache_setup['data_root'],
        version_hash=temp_cache_setup['version_hash'],
        use_cpu_cache=True,
        use_disk_cache=True,
        max_cpu_memory_percent=max_memory_percent,
        enable_cpu_validation=enable_cpu_validation,
        enable_disk_validation=enable_disk_validation,
        dataset_class_name=dataset_class_name,
        version_dict=version_dict
    )
    
    # Verify CPU cache parameters
    assert cache.cpu_cache.max_memory_percent == max_memory_percent
    assert cache.cpu_cache.enable_validation == enable_cpu_validation
    
    # Verify disk cache parameters
    assert cache.disk_cache.version_hash == temp_cache_setup['version_hash']
    assert cache.disk_cache.enable_validation == enable_disk_validation
    assert cache.disk_cache.dataset_class_name == dataset_class_name
    assert cache.disk_cache.version_dict == version_dict


def test_cache_initialization_directory_structure(temp_cache_setup):
    """Test that cache initialization creates proper directory structure."""
    cache = CombinedDatasetCache(
        data_root=temp_cache_setup['data_root'],
        version_hash=temp_cache_setup['version_hash'],
        use_disk_cache=True
    )
    
    # Verify disk cache directory structure exists
    expected_cache_dir = f"{temp_cache_setup['data_root']}_cache"
    expected_version_dir = os.path.join(expected_cache_dir, temp_cache_setup['version_hash'])
    
    assert os.path.exists(expected_cache_dir)
    assert os.path.exists(expected_version_dir)
    assert cache.disk_cache.cache_dir == expected_cache_dir
    assert cache.disk_cache.version_dir == expected_version_dir


def test_cache_initialization_default_parameters(temp_cache_setup):
    """Test cache initialization with default parameters."""
    cache = CombinedDatasetCache(
        data_root=temp_cache_setup['data_root'],
        version_hash=temp_cache_setup['version_hash']
    )
    
    # Verify default configuration
    assert cache.use_cpu_cache is True  # Default
    assert cache.use_disk_cache is True  # Default
    
    # Verify components are initialized with defaults
    assert cache.cpu_cache is not None
    assert cache.disk_cache is not None
    assert cache.cpu_cache.max_memory_percent == 80.0  # Default
    assert cache.cpu_cache.enable_validation is False  # Default 
    assert cache.disk_cache.enable_validation is False  # Default


def test_cache_initialization_with_optional_metadata(temp_cache_setup):
    """Test cache initialization with optional metadata parameters."""
    dataset_class_name = 'TestDataset'
    version_dict = {
        'transforms': ['normalize', 'resize'],
        'split': 'train',
        'version': '1.2.3'
    }
    
    cache = CombinedDatasetCache(
        data_root=temp_cache_setup['data_root'],
        version_hash=temp_cache_setup['version_hash'],
        dataset_class_name=dataset_class_name,
        version_dict=version_dict
    )
    
    # Verify metadata is properly stored in disk cache
    if cache.disk_cache:
        assert cache.disk_cache.dataset_class_name == dataset_class_name
        assert cache.disk_cache.version_dict == version_dict


def test_cache_initialization_without_optional_metadata(temp_cache_setup):
    """Test cache initialization without optional metadata parameters."""
    cache = CombinedDatasetCache(
        data_root=temp_cache_setup['data_root'],
        version_hash=temp_cache_setup['version_hash']
    )
    
    # Verify None values are handled properly
    if cache.disk_cache:
        assert cache.disk_cache.dataset_class_name is None
        assert cache.disk_cache.version_dict is None


def test_cache_initialization_extreme_memory_limits(temp_cache_setup):
    """Test cache initialization with extreme memory limit values."""
    # Test very low memory limit
    cache_low = CombinedDatasetCache(
        data_root=temp_cache_setup['data_root'],
        version_hash=temp_cache_setup['version_hash'],
        max_cpu_memory_percent=1.0  # Very low
    )
    
    assert cache_low.cpu_cache.max_memory_percent == 1.0
    
    # Test very high memory limit  
    cache_high = CombinedDatasetCache(
        data_root=temp_cache_setup['data_root'],
        version_hash=temp_cache_setup['version_hash'],
        max_cpu_memory_percent=99.0  # Very high
    )
    
    assert cache_high.cpu_cache.max_memory_percent == 99.0


def test_cache_initialization_validation_combinations(temp_cache_setup):
    """Test all combinations of validation enable/disable for both cache types."""
    validation_combinations = [
        (True, True, 'both_validation_enabled'),
        (True, False, 'cpu_validation_only'),
        (False, True, 'disk_validation_only'),
        (False, False, 'no_validation')
    ]
    
    for cpu_val, disk_val, description in validation_combinations:
        cache = CombinedDatasetCache(
            data_root=temp_cache_setup['data_root'],
            version_hash=temp_cache_setup['version_hash'],
            enable_cpu_validation=cpu_val,
            enable_disk_validation=disk_val
        )
        
        # Verify validation settings
        if cache.cpu_cache:
            assert cache.cpu_cache.enable_validation == cpu_val, f"CPU validation mismatch for {description}"
        if cache.disk_cache:
            assert cache.disk_cache.enable_validation == disk_val, f"Disk validation mismatch for {description}"