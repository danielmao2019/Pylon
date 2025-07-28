"""Tests for LevirCdDataset cache version discrimination."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset

# Generate hash discrimination tests using the centralized template fixture
@pytest.fixture
def _hash_tests(hash_discrimination_tests):
    return hash_discrimination_tests(
        dataset_class=LevirCdDataset,
        data_root_fixture_name='levir_cd_data_root', 
        splits=['train', 'val', 'test']
    )

def test_levir_cd_same_parameters_same_hash(_hash_tests, request):
    """Test that identical parameters produce identical hashes."""
    _hash_tests['test_same_parameters_same_hash'](request)

def test_levir_cd_different_split_different_hash(_hash_tests, request):
    """Test that different splits produce different hashes.""" 
    _hash_tests['test_different_split_different_hash'](request)

def test_levir_cd_hash_format(_hash_tests, request):
    """Test that hash is in correct format."""
    _hash_tests['test_hash_format'](request)

def test_levir_cd_comprehensive_no_hash_collisions(_hash_tests, request):
    """Test that different configurations produce unique hashes."""
    _hash_tests['test_comprehensive_no_hash_collisions'](request)


def test_parameters_that_dont_affect_version_hash(levir_cd_data_root):
    """Test that cache/processing parameters don't affect version hash (correct caching behavior)."""
    
    base_args = {
        'data_root': levir_cd_data_root,
        'split': 'train',
    }
    
    # Test parameters that affect processing but not dataset content
    parameter_variants = [
        ('base_seed', 42),  # Affects randomness, not content
        ('max_cache_memory_percent', 50.0),  # Affects caching, not content
        ('use_cpu_cache', False),  # Affects caching, not content
        ('use_disk_cache', False),  # Affects caching, not content
    ]
    
    dataset1 = LevirCdDataset(**base_args)
    
    for param_name, new_value in parameter_variants:
        modified_args = base_args.copy()
        modified_args[param_name] = new_value
        dataset2 = LevirCdDataset(**modified_args)
        
        assert dataset1.get_cache_version_hash() == dataset2.get_cache_version_hash(), \
            f"Processing parameter {param_name} should not affect cache version hash (content unchanged)"


