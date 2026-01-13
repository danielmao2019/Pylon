import pytest
import tempfile
import torch
import copy
import os
from typing import Dict, Any
from data.cache.combined_dataset_cache import CombinedDatasetCache


@pytest.fixture
def temp_cache_setup():
    """Create temporary directory setup for combined cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield {
            'data_root': temp_dir,
            'version_hash': 'test_combined_cache_v123'
        }


@pytest.fixture
def sample_datapoint():
    """Create a sample datapoint for combined cache testing."""
    return {
        'inputs': {
            'image': torch.randn(3, 64, 64, dtype=torch.float32),
            'features': torch.randn(256, dtype=torch.float32)
        },
        'labels': {
            'mask': torch.randint(0, 2, (64, 64), dtype=torch.long),
            'class_id': torch.tensor([1], dtype=torch.long)
        },
        'meta_info': {
            'path': '/test/sample.jpg',
            'index': 0,
            'dataset': 'test_dataset'
        }
    }


@pytest.fixture
def different_datapoint():
    """Create a different datapoint for testing cache hierarchy."""
    return {
        'inputs': {
            'image': torch.randn(3, 64, 64, dtype=torch.float32) + 10.0,  # Different values
            'features': torch.randn(256, dtype=torch.float32) + 5.0
        },
        'labels': {
            'mask': torch.randint(0, 2, (64, 64), dtype=torch.long),
            'class_id': torch.tensor([2], dtype=torch.long)  # Different class
        },
        'meta_info': {
            'path': '/test/different.jpg',
            'index': 1,
            'dataset': 'test_dataset'
        }
    }


@pytest.fixture
def cache_config_factory(temp_cache_setup):
    """Factory fixture for creating caches with different configurations."""
    def _create_cache(
        use_cpu_cache: bool = True,
        use_disk_cache: bool = True,
        max_cpu_memory_percent: float = 80.0,
        enable_cpu_validation: bool = False,
        enable_disk_validation: bool = False,
        dataset_class_name: str = 'TestDataset',
        version_dict: Dict[str, Any] = None
    ):
        return CombinedDatasetCache(
            data_root=temp_cache_setup['data_root'],
            version_hash=temp_cache_setup['version_hash'],
            use_cpu_cache=use_cpu_cache,
            use_disk_cache=use_disk_cache,
            max_cpu_memory_percent=max_cpu_memory_percent,
            enable_cpu_validation=enable_cpu_validation,
            enable_disk_validation=enable_disk_validation,
            dataset_class_name=dataset_class_name,
            version_dict=version_dict or {'test_param': 'test_value'}
        )
    return _create_cache


@pytest.fixture
def all_cache_configurations(cache_config_factory):
    """Fixture providing all combinations of cache enable/disable configurations."""
    return [
        # (use_cpu_cache, use_disk_cache, description)
        (True, True, 'both_enabled'),
        (True, False, 'cpu_only'),
        (False, True, 'disk_only'),
        (False, False, 'both_disabled')
    ]


@pytest.fixture
def cache_with_both_enabled(cache_config_factory):
    """Create a cache with both CPU and disk enabled for standard testing."""
    return cache_config_factory(
        use_cpu_cache=True,
        use_disk_cache=True
    )


@pytest.fixture
def cache_cpu_only(cache_config_factory):
    """Create a cache with only CPU enabled."""
    return cache_config_factory(
        use_cpu_cache=True,
        use_disk_cache=False
    )


@pytest.fixture
def cache_disk_only(cache_config_factory):
    """Create a cache with only disk enabled."""
    return cache_config_factory(
        use_cpu_cache=False,
        use_disk_cache=True
    )


@pytest.fixture
def cache_both_disabled(cache_config_factory):
    """Create a cache with both CPU and disk disabled."""
    return cache_config_factory(
        use_cpu_cache=False,
        use_disk_cache=False
    )


@pytest.fixture
def hierarchical_test_setup(cache_config_factory, sample_datapoint, different_datapoint):
    """Setup for testing hierarchical cache behavior with pre-populated data."""
    def _setup_hierarchy(cpu_data_idx=None, disk_data_idx=None):
        cache = cache_config_factory(use_cpu_cache=True, use_disk_cache=True)

        # Pre-populate CPU cache if specified
        if cpu_data_idx is not None:
            cpu_cache_filepath = os.path.join(cache.version_dir, f"{cpu_data_idx}.pt")
            cache.cpu_cache.put(
                sample_datapoint,
                cache_filepath=cpu_cache_filepath,
            )

        # Pre-populate disk cache if specified
        if disk_data_idx is not None:
            cache.disk_cache.put(
                different_datapoint,
                cache_filepath=os.path.join(cache.version_dir, f"{disk_data_idx}.pt"),
            )

        return cache

    return _setup_hierarchy


@pytest.fixture
def device_test_scenarios():
    """Provide various device configurations for testing device parameter handling."""
    return [
        None,  # Default (no device specified)
        'cpu',
        'cuda' if torch.cuda.is_available() else 'cpu',  # Use CUDA if available, fallback to CPU
    ]


@pytest.fixture
def make_datapoint_factory():
    """Factory to create unique datapoints for testing."""
    def _make_datapoint(idx: int, add_variation: bool = True):
        base_value = float(idx) if add_variation else 0.0

        return {
            'inputs': {
                'image': torch.randn(3, 64, 64, dtype=torch.float32) + base_value,
                'features': torch.randn(256, dtype=torch.float32) + base_value
            },
            'labels': {
                'mask': torch.randint(0, 2, (64, 64), dtype=torch.long),
                'class_id': torch.tensor([idx % 10], dtype=torch.long)
            },
            'meta_info': {
                'path': f'/test/sample_{idx}.jpg',
                'index': idx,
                'dataset': 'test_dataset'
            }
        }

    return _make_datapoint


@pytest.fixture
def large_datapoint():
    """Create a larger datapoint for memory and performance testing."""
    return {
        'inputs': {
            'image': torch.randn(3, 512, 512, dtype=torch.float32),  # Larger image
            'features': torch.randn(2048, dtype=torch.float32),       # Larger feature vector
            'extra_data': torch.randn(1000, 1000, dtype=torch.float32)  # Additional large tensor
        },
        'labels': {
            'mask': torch.randint(0, 10, (512, 512), dtype=torch.long),
            'class_id': torch.tensor([5], dtype=torch.long)
        },
        'meta_info': {
            'path': '/test/large_sample.jpg',
            'index': 999,
            'dataset': 'large_test_dataset',
            'additional_info': 'This is a large datapoint for testing'
        }
    }
