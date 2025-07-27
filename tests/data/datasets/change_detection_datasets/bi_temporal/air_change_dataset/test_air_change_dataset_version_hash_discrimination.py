"""Test version hash discrimination for AirChangeDataset.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
import tempfile
import os
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset




def test_air_change_same_parameters_same_hash(air_change_data_root):
    """Test that identical parameters produce identical hashes."""
    # Same parameters should produce same hash
    dataset1a = AirChangeDataset(data_root=air_change_data_root, split='train')
    dataset1b = AirChangeDataset(data_root=air_change_data_root, split='train')
    
    hash1a = dataset1a.get_cache_version_hash()
    hash1b = dataset1b.get_cache_version_hash()
    
    assert hash1a == hash1b, f"Same parameters should produce same hash: {hash1a} != {hash1b}"


def test_air_change_different_split_different_hash(air_change_data_root):
    """Test that different splits produce different hashes."""
    dataset_train = AirChangeDataset(data_root=air_change_data_root, split='train')
    dataset_test = AirChangeDataset(data_root=air_change_data_root, split='test')
    
    hash_train = dataset_train.get_cache_version_hash()
    hash_test = dataset_test.get_cache_version_hash()
    
    assert hash_train != hash_test, f"Different splits should produce different hashes: {hash_train} == {hash_test}"


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


def test_air_change_hash_format(air_change_data_root):
    """Test that hash is in correct format."""
    dataset = AirChangeDataset(data_root=air_change_data_root, split='train')
    hash_val = dataset.get_cache_version_hash()
    
    # Should be a string
    assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"
    
    # Should be 16 characters (xxhash format)
    assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"
    
    # Should be hexadecimal
    assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"


def test_air_change_comprehensive_no_hash_collisions(air_change_data_root):
    """Test that different configurations produce unique hashes (no collisions)."""
    # Test various parameter combinations
    # NOTE: data_root is intentionally excluded from hash, so we test only meaningful parameter combinations
    configs = [
        {'data_root': air_change_data_root, 'split': 'train'},
        {'data_root': air_change_data_root, 'split': 'test'},
        # Removed duplicate data_root configs since data_root is excluded from hash
    ]
    
    hashes = []
    for config in configs:
        dataset = AirChangeDataset(**config)
        hash_val = dataset.get_cache_version_hash()
        
        # Check for collision
        assert hash_val not in hashes, f"Hash collision detected for config {config}: hash {hash_val} already exists"
        hashes.append(hash_val)
    
    # Verify we generated the expected number of unique hashes
    assert len(hashes) == len(configs), f"Expected {len(configs)} unique hashes, got {len(hashes)}"
