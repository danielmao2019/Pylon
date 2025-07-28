"""Test version hash discrimination for CelebADataset.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset




def test_celeb_a_same_parameters_same_hash(celeb_a_data_root):
    """Test that identical parameters produce identical hashes."""
    # Same parameters should produce same hash
    dataset1a = CelebADataset(data_root=celeb_a_data_root, split='train', use_landmarks=True)
    dataset1b = CelebADataset(data_root=celeb_a_data_root, split='train', use_landmarks=True)
    
    hash1a = dataset1a.get_cache_version_hash()
    hash1b = dataset1b.get_cache_version_hash()
    
    assert hash1a == hash1b, f"Same parameters should produce same hash: {hash1a} != {hash1b}"


def test_celeb_a_different_use_landmarks_different_hash(celeb_a_data_root):
    """Test that different use_landmarks settings produce different hashes."""
    dataset_with_landmarks = CelebADataset(data_root=celeb_a_data_root, split='train', use_landmarks=True)
    dataset_without_landmarks = CelebADataset(data_root=celeb_a_data_root, split='train', use_landmarks=False)
    
    hash_with = dataset_with_landmarks.get_cache_version_hash()
    hash_without = dataset_without_landmarks.get_cache_version_hash()
    
    assert hash_with != hash_without, f"Different use_landmarks should produce different hashes: {hash_with} == {hash_without}"


def test_celeb_a_different_split_different_hash(celeb_a_data_root):
    """Test that different splits produce different hashes."""
    dataset_train = CelebADataset(data_root=celeb_a_data_root, split='train', use_landmarks=False)
    dataset_val = CelebADataset(data_root=celeb_a_data_root, split='val', use_landmarks=False)
    
    hash_train = dataset_train.get_cache_version_hash()
    hash_val = dataset_val.get_cache_version_hash()
    
    assert hash_train != hash_val, f"Different splits should produce different hashes: {hash_train} == {hash_val}"


def test_celeb_a_hash_format(celeb_a_data_root):
    """Test that hash is in correct format."""
    dataset = CelebADataset(data_root=celeb_a_data_root, split='train', use_landmarks=True)
    hash_val = dataset.get_cache_version_hash()
    
    # Should be a string
    assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"
    
    # Should be 16 characters (xxhash format)
    assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"
    
    # Should be hexadecimal
    assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"


def test_celeb_a_comprehensive_no_hash_collisions(celeb_a_data_root):
    """Test that different configurations produce unique hashes (no collisions)."""
    # Test various parameter combinations
    # NOTE: data_root is intentionally excluded from hash, so we test only meaningful parameter combinations
    configs = [
        {'data_root': celeb_a_data_root, 'split': 'train', 'use_landmarks': True},
        {'data_root': celeb_a_data_root, 'split': 'train', 'use_landmarks': False},
        {'data_root': celeb_a_data_root, 'split': 'val', 'use_landmarks': True},
        {'data_root': celeb_a_data_root, 'split': 'val', 'use_landmarks': False},
        # Removed different data_root configs since data_root is excluded from hash
    ]
    
    hashes = []
    for config in configs:
        dataset = CelebADataset(**config)
        hash_val = dataset.get_cache_version_hash()
        
        # Check for collision
        assert hash_val not in hashes, f"Hash collision detected for config {config}: hash {hash_val} already exists"
        hashes.append(hash_val)
    
    # Verify we generated the expected number of unique hashes
    assert len(hashes) == len(configs), f"Expected {len(configs)} unique hashes, got {len(hashes)}"
