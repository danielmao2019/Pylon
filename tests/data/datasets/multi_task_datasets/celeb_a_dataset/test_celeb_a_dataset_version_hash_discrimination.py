"""Test version hash discrimination for CelebADataset.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset

# Generate hash discrimination tests using the centralized template fixture
# Note: CelebA has specific requirements and additional parameters
@pytest.fixture
def _hash_tests(hash_discrimination_tests):
    # For now, use basic template - CelebA's additional_params feature needs template enhancement
    return hash_discrimination_tests(
        dataset_class=CelebADataset,
        data_root_fixture_name='celeb_a_data_root', 
        splits=['train', 'val']
    )

def test_celeb_a_same_parameters_same_hash(_hash_tests, request):
    """Test that identical parameters produce identical hashes."""
    _hash_tests['test_same_parameters_same_hash'](request)

def test_celeb_a_different_split_different_hash(_hash_tests, request):
    """Test that different splits produce different hashes.""" 
    _hash_tests['test_different_split_different_hash'](request)

def test_celeb_a_hash_format(_hash_tests, request):
    """Test that hash is in correct format."""
    _hash_tests['test_hash_format'](request)

def test_celeb_a_comprehensive_no_hash_collisions(_hash_tests, request):
    """Test that different configurations produce unique hashes."""
    _hash_tests['test_comprehensive_no_hash_collisions'](request)


def test_celeb_a_different_use_landmarks_different_hash(celeb_a_data_root):
    """Test that different use_landmarks settings produce different hashes."""
    dataset_with_landmarks = CelebADataset(data_root=celeb_a_data_root, split='train', use_landmarks=True)
    dataset_without_landmarks = CelebADataset(data_root=celeb_a_data_root, split='train', use_landmarks=False)
    
    hash_with = dataset_with_landmarks.get_cache_version_hash()
    hash_without = dataset_without_landmarks.get_cache_version_hash()
    
    assert hash_with != hash_without, f"Different use_landmarks should produce different hashes: {hash_with} == {hash_without}"
