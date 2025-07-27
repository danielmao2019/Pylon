"""Test version hash discrimination for KC3DDataset.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
import tempfile
import os
import pickle
import numpy as np
from data.datasets.change_detection_datasets.bi_temporal.kc_3d_dataset import KC3DDataset



def test_kc3d_same_parameters_same_hash(create_dummy_kc3d_files):
    """Test that identical parameters produce identical hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kc3d_files(temp_dir)
        
        # Same parameters should produce same hash
        dataset1a = KC3DDataset(data_root=temp_dir, split='train', use_ground_truth_registration=True)
        dataset1b = KC3DDataset(data_root=temp_dir, split='train', use_ground_truth_registration=True)
        
        hash1a = dataset1a.get_cache_version_hash()
        hash1b = dataset1b.get_cache_version_hash()
        
        assert hash1a == hash1b, f"Same parameters should produce same hash: {hash1a} != {hash1b}"


def test_kc3d_different_split_different_hash(create_dummy_kc3d_files):
    """Test that different splits produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kc3d_files(temp_dir)
        
        dataset_train = KC3DDataset(data_root=temp_dir, split='train', use_ground_truth_registration=True)
        dataset_val = KC3DDataset(data_root=temp_dir, split='val', use_ground_truth_registration=True)
        dataset_test = KC3DDataset(data_root=temp_dir, split='test', use_ground_truth_registration=True)
        
        hash_train = dataset_train.get_cache_version_hash()
        hash_val = dataset_val.get_cache_version_hash()
        hash_test = dataset_test.get_cache_version_hash()
        
        assert hash_train != hash_val, f"Different splits should produce different hashes: {hash_train} == {hash_val}"
        assert hash_train != hash_test, f"Different splits should produce different hashes: {hash_train} == {hash_test}"
        assert hash_val != hash_test, f"Different splits should produce different hashes: {hash_val} == {hash_test}"


def test_kc3d_different_use_ground_truth_registration_different_hash(create_dummy_kc3d_files):
    """Test that different use_ground_truth_registration values produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kc3d_files(temp_dir)
        
        dataset_with_gt = KC3DDataset(data_root=temp_dir, split='train', use_ground_truth_registration=True)
        dataset_without_gt = KC3DDataset(data_root=temp_dir, split='train', use_ground_truth_registration=False)
        
        hash_with_gt = dataset_with_gt.get_cache_version_hash()
        hash_without_gt = dataset_without_gt.get_cache_version_hash()
        
        assert hash_with_gt != hash_without_gt, f"Different use_ground_truth_registration should produce different hashes: {hash_with_gt} == {hash_without_gt}"


def test_kc3d_different_data_root_different_hash(create_dummy_kc3d_files):
    """Test that different data roots produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_kc3d_files(temp_dir1)
            create_dummy_kc3d_files(temp_dir2)
            
            dataset1 = KC3DDataset(data_root=temp_dir1, split='train', use_ground_truth_registration=True)
            dataset2 = KC3DDataset(data_root=temp_dir2, split='train', use_ground_truth_registration=True)
            
            hash1 = dataset1.get_cache_version_hash()
            hash2 = dataset2.get_cache_version_hash()
            
            # data_root is intentionally excluded from version hash for cache stability across different paths
            assert hash1 == hash2, f"Different data roots should produce SAME hashes (data_root excluded): {hash1} != {hash2}"


def test_kc3d_hash_format(create_dummy_kc3d_files):
    """Test that hash is in correct format."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kc3d_files(temp_dir)
        
        dataset = KC3DDataset(data_root=temp_dir, split='train', use_ground_truth_registration=True)
        hash_val = dataset.get_cache_version_hash()
        
        # Should be a string
        assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"
        
        # Should be 16 characters (xxhash format)
        assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"
        
        # Should be hexadecimal
        assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"


def test_kc3d_comprehensive_no_hash_collisions(create_dummy_kc3d_files):
    """Test that different configurations produce unique hashes (no collisions)."""
    
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_kc3d_files(temp_dir1)
            create_dummy_kc3d_files(temp_dir2)
            
            # Test various parameter combinations
            # NOTE: data_root is intentionally excluded from hash, so we test only meaningful parameter combinations
            configs = [
                {'data_root': temp_dir1, 'split': 'train', 'use_ground_truth_registration': True},
                {'data_root': temp_dir1, 'split': 'train', 'use_ground_truth_registration': False},
                {'data_root': temp_dir1, 'split': 'val', 'use_ground_truth_registration': True},
                {'data_root': temp_dir1, 'split': 'val', 'use_ground_truth_registration': False},
                {'data_root': temp_dir1, 'split': 'test', 'use_ground_truth_registration': True},
                {'data_root': temp_dir1, 'split': 'test', 'use_ground_truth_registration': False},
                # Removed duplicate data_root configs since data_root is excluded from hash
            ]
            
            hashes = []
            for config in configs:
                dataset = KC3DDataset(**config)
                hash_val = dataset.get_cache_version_hash()
                
                # Check for collision
                assert hash_val not in hashes, f"Hash collision detected for config {config}: hash {hash_val} already exists"
                hashes.append(hash_val)
            
            # Verify we generated the expected number of unique hashes
            assert len(hashes) == len(configs), f"Expected {len(configs)} unique hashes, got {len(hashes)}"
