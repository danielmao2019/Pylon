"""Test version hash discrimination for xView2Dataset.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
import tempfile
import os
from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import xView2Dataset



def test_xview2_same_parameters_same_hash(create_dummy_xview2_files):
    """Test that identical parameters produce identical hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_xview2_files(temp_dir)
        
        # Same parameters should produce same hash
        dataset1a = xView2Dataset(data_root=temp_dir, split='train')
        dataset1b = xView2Dataset(data_root=temp_dir, split='train')
        
        hash1a = dataset1a.get_cache_version_hash()
        hash1b = dataset1b.get_cache_version_hash()
        
        assert hash1a == hash1b, f"Same parameters should produce same hash: {hash1a} != {hash1b}"


def test_xview2_different_split_different_hash(create_dummy_xview2_files):
    """Test that different splits produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_xview2_files(temp_dir)
        
        dataset_train = xView2Dataset(data_root=temp_dir, split='train')
        dataset_test = xView2Dataset(data_root=temp_dir, split='test')
        dataset_hold = xView2Dataset(data_root=temp_dir, split='hold')
        
        hash_train = dataset_train.get_cache_version_hash()
        hash_test = dataset_test.get_cache_version_hash()
        hash_hold = dataset_hold.get_cache_version_hash()
        
        assert hash_train != hash_test, f"Different splits should produce different hashes: {hash_train} == {hash_test}"
        assert hash_train != hash_hold, f"Different splits should produce different hashes: {hash_train} == {hash_hold}"
        assert hash_test != hash_hold, f"Different splits should produce different hashes: {hash_test} == {hash_hold}"


def test_xview2_different_data_root_different_hash(create_dummy_xview2_files):
    """Test that different data roots produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_xview2_files(temp_dir1)
            create_dummy_xview2_files(temp_dir2)
            
            dataset1 = xView2Dataset(data_root=temp_dir1, split='train')
            dataset2 = xView2Dataset(data_root=temp_dir2, split='train')
            
            hash1 = dataset1.get_cache_version_hash()
            hash2 = dataset2.get_cache_version_hash()
            
            assert hash1 != hash2, f"Different data roots should produce different hashes: {hash1} == {hash2}"


def test_xview2_hash_format(create_dummy_xview2_files):
    """Test that hash is in correct format."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_xview2_files(temp_dir)
        
        dataset = xView2Dataset(data_root=temp_dir, split='train')
        hash_val = dataset.get_cache_version_hash()
        
        # Should be a string
        assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"
        
        # Should be 16 characters (xxhash format)
        assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"
        
        # Should be hexadecimal
        assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"


def test_xview2_comprehensive_no_hash_collisions(create_dummy_xview2_files):
    """Test that different configurations produce unique hashes (no collisions)."""
    
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_xview2_files(temp_dir1)
            create_dummy_xview2_files(temp_dir2)
            
            # Test various parameter combinations
            configs = [
                {'data_root': temp_dir1, 'split': 'train'},
                {'data_root': temp_dir1, 'split': 'test'},
                {'data_root': temp_dir1, 'split': 'hold'},
                {'data_root': temp_dir2, 'split': 'train'},
                {'data_root': temp_dir2, 'split': 'test'},
                {'data_root': temp_dir2, 'split': 'hold'},
            ]
            
            hashes = []
            for config in configs:
                dataset = xView2Dataset(**config)
                hash_val = dataset.get_cache_version_hash()
                
                # Check for collision
                assert hash_val not in hashes, f"Hash collision detected for config {config}: hash {hash_val} already exists"
                hashes.append(hash_val)
            
            # Verify we generated the expected number of unique hashes
            assert len(hashes) == len(configs), f"Expected {len(configs)} unique hashes, got {len(hashes)}"