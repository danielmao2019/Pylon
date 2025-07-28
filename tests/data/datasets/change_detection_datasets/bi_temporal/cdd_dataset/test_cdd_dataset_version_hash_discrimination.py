"""Test version hash discrimination for CDDDataset.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.cdd_dataset import CDDDataset

# Generate hash discrimination tests using the centralized template fixture
@pytest.fixture
def _hash_tests(hash_discrimination_tests):
    return hash_discrimination_tests(
        dataset_class=CDDDataset,
        data_root_fixture_name='cdd_data_root', 
        splits=['train', 'val', 'test']
    )

def test_cdd_same_parameters_same_hash(_hash_tests, request):
    """Test that identical parameters produce identical hashes."""
    _hash_tests['test_same_parameters_same_hash'](request)

def test_cdd_different_split_different_hash(_hash_tests, request):
    """Test that different splits produce different hashes.""" 
    _hash_tests['test_different_split_different_hash'](request)

def test_cdd_hash_format(_hash_tests, request):
    """Test that hash is in correct format."""
    _hash_tests['test_hash_format'](request)

def test_cdd_comprehensive_no_hash_collisions(_hash_tests, request):
    """Test that different configurations produce unique hashes."""
    _hash_tests['test_comprehensive_no_hash_collisions'](request)


def test_cdd_same_data_root_same_hash(cdd_data_root):
    """Test that same data roots produce same hashes (data_root excluded from hash)."""
    # data_root is intentionally excluded from hash for cache stability
    dataset1 = CDDDataset(data_root=cdd_data_root, split='train')
    dataset2 = CDDDataset(data_root=cdd_data_root, split='train')
    
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    assert hash1 == hash2, f"Same data roots should produce same hashes: {hash1} != {hash2}"
