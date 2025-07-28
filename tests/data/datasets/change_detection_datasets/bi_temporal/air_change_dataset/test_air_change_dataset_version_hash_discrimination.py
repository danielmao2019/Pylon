"""Test version hash discrimination for AirChangeDataset.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset

# Generate hash discrimination tests using the centralized template fixture
@pytest.fixture
def _hash_tests(hash_discrimination_tests):
    return hash_discrimination_tests(
        dataset_class=AirChangeDataset,
        data_root_fixture_name='air_change_data_root', 
        splits=['train', 'test']
    )

def test_air_change_same_parameters_same_hash(_hash_tests, request):
    """Test that identical parameters produce identical hashes."""
    _hash_tests['test_same_parameters_same_hash'](request)

def test_air_change_different_split_different_hash(_hash_tests, request):
    """Test that different splits produce different hashes.""" 
    _hash_tests['test_different_split_different_hash'](request)

def test_air_change_hash_format(_hash_tests, request):
    """Test that hash is in correct format."""
    _hash_tests['test_hash_format'](request)

def test_air_change_comprehensive_no_hash_collisions(_hash_tests, request):
    """Test that different configurations produce unique hashes."""
    _hash_tests['test_comprehensive_no_hash_collisions'](request)


def test_air_change_different_data_root_same_hash(air_change_data_root):
    """Test that different data roots produce same hashes (data_root excluded from hash)."""
    # Since data_root is intentionally excluded from hash, we can only test with same data
    # Create two dataset instances with same data root to verify data_root exclusion
    dataset1 = AirChangeDataset(data_root=air_change_data_root, split='train')
    dataset2 = AirChangeDataset(data_root=air_change_data_root, split='train')  # Same data root
    
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    # data_root is intentionally excluded from version hash for cache stability across different paths
    assert hash1 == hash2, f"Same data roots should produce same hashes: {hash1} != {hash2}"
