from typing import Any, Dict, Tuple
import pytest
import torch
from data.datasets.base_dataset import BaseDataset
from data.transforms.compose import Compose
from data.transforms.random_noise import RandomNoise


def pytest_addoption(parser):
    """Add custom command line options for dataset testing."""
    parser.addoption(
        "--samples",
        action="store",
        default=None,
        type=int,
        help="Maximum number of datapoints to test per dataset (default: test full dataset)"
    )


@pytest.fixture
def max_samples(request):
    """Fixture to get the maximum number of samples to test from command line."""
    return request.config.getoption("--samples")


@pytest.fixture
def get_samples_to_test():
    """
    Fixture that provides helper function to determine how many samples to test.
    """
    def _get_samples_to_test(dataset_length: int, max_samples: int = None) -> int:
        """
        Helper function to determine how many samples to test based on command line args.

        Args:
            dataset_length: Total number of samples in the dataset
            max_samples: Value from --samples command line argument (can be None)

        Returns:
            Number of samples to test (full dataset if --samples not provided)
        """
        if max_samples is not None:
            return min(dataset_length, max_samples)
        else:
            return dataset_length
    return _get_samples_to_test


@pytest.fixture
def SampleDataset():
    """A minimal dataset implementation for testing BaseDataset functionality."""
    class _SampleDataset(BaseDataset):
        SPLIT_OPTIONS = ['train', 'val', 'test', 'weird']
        INPUT_NAMES = ['input']
        LABEL_NAMES = ['label']

        def _init_annotations(self) -> None:
            # all splits are the same
            self.annotations = list(range(100))

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
        ]:
            # Create a random tensor for testing transforms
            # Use the annotation index as the random seed for reproducibility
            torch.manual_seed(self.annotations[idx])
            tensor = torch.randn(3, 32, 32)  # Random noise
            return {'input': tensor}, {'label': self.annotations[idx]}, {}

    return _SampleDataset


@pytest.fixture
def random_transforms():
    """Random transforms for testing cache behavior with randomization."""
    noise = RandomNoise(std=0.2)  # Add significant noise for visible effect

    return {
        'class': Compose,
        'args': {
            'transforms': [
                (noise, [('inputs', 'input')]),
            ]
        }
    }


# =============================================================================
# Version Hash Discrimination Test Templates
# =============================================================================

def create_hash_discrimination_tests(dataset_class: type, data_root_fixture_name: str, splits: list = None):
    """
    Factory function to generate version hash discrimination tests for any dataset class.
    
    This eliminates code duplication across 12+ dataset test files that all implement
    identical hash discrimination test patterns.
    
    Args:
        dataset_class: The dataset class to test (e.g., AirChangeDataset, CDDDataset)
        data_root_fixture_name: Name of the data root fixture (e.g., 'air_change_data_root')
        splits: List of splits to test. If None, defaults to ['train', 'test']
        
    Returns:
        Dictionary of test functions that can be injected into test modules
    """
    if splits is None:
        splits = ['train', 'test']
    
    def test_same_parameters_same_hash(request):
        """Test that identical parameters produce identical hashes."""
        data_root = request.getfixturevalue(data_root_fixture_name)
        
        # Same parameters should produce same hash
        dataset1a = dataset_class(data_root=data_root, split=splits[0])
        dataset1b = dataset_class(data_root=data_root, split=splits[0])
        
        hash1a = dataset1a.get_cache_version_hash()
        hash1b = dataset1b.get_cache_version_hash()
        
        assert hash1a == hash1b, f"Same parameters should produce same hash: {hash1a} != {hash1b}"

    def test_different_split_different_hash(request):
        """Test that different splits produce different hashes."""
        data_root = request.getfixturevalue(data_root_fixture_name)
        
        if len(splits) < 2:
            pytest.skip(f"Dataset {dataset_class.__name__} only has one split: {splits}")
        
        dataset1 = dataset_class(data_root=data_root, split=splits[0])
        dataset2 = dataset_class(data_root=data_root, split=splits[1])
        
        hash1 = dataset1.get_cache_version_hash()
        hash2 = dataset2.get_cache_version_hash()
        
        assert hash1 != hash2, f"Different splits should produce different hashes: {hash1} == {hash2}"

    def test_hash_format(request):
        """Test that hash is in correct format."""
        data_root = request.getfixturevalue(data_root_fixture_name)
        
        dataset = dataset_class(data_root=data_root, split=splits[0])
        hash_val = dataset.get_cache_version_hash()
        
        # Should be a string
        assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"
        
        # Should be 16 characters (xxhash format)
        assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"
        
        # Should be hexadecimal
        assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"

    def test_comprehensive_no_hash_collisions(request):
        """Test that different configurations produce unique hashes (no collisions)."""
        data_root = request.getfixturevalue(data_root_fixture_name)
        
        # Test various parameter combinations
        # NOTE: data_root is intentionally excluded from hash, so we test only meaningful parameter combinations
        configs = [{'data_root': data_root, 'split': split} for split in splits]
        
        hashes = []
        for config in configs:
            dataset = dataset_class(**config)
            hash_val = dataset.get_cache_version_hash()
            
            # Check for collision
            assert hash_val not in hashes, f"Hash collision detected for config {config}: hash {hash_val} already exists"
            hashes.append(hash_val)
        
        # Verify we generated the expected number of unique hashes
        assert len(hashes) == len(configs), f"Expected {len(configs)} unique hashes, got {len(hashes)}"

    return {
        'test_same_parameters_same_hash': test_same_parameters_same_hash,
        'test_different_split_different_hash': test_different_split_different_hash, 
        'test_hash_format': test_hash_format,
        'test_comprehensive_no_hash_collisions': test_comprehensive_no_hash_collisions,
    }


@pytest.fixture 
def hash_discrimination_tests():
    """Fixture providing the hash discrimination test creation function."""
    return create_hash_discrimination_tests


@pytest.fixture
def mnist_data_root():
    """Fixture that provides MNIST data path."""
    return "./data/datasets/soft_links/MNIST"
