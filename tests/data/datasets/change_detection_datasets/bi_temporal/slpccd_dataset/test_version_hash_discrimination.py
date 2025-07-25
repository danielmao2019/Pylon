"""Test version hash discrimination for SLPCCDDataset.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
import tempfile
import os
from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset



def test_slpccd_same_parameters_same_hash(create_dummy_slpccd_files):
    """Test that identical parameters produce identical hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_slpccd_files(temp_dir)
        
        # Same parameters should produce same hash
        dataset1a = SLPCCDDataset(data_root=temp_dir, split='train', num_points=8192, random_subsample=True)
        dataset1b = SLPCCDDataset(data_root=temp_dir, split='train', num_points=8192, random_subsample=True)
        
        hash1a = dataset1a.get_cache_version_hash()
        hash1b = dataset1b.get_cache_version_hash()
        
        assert hash1a == hash1b, f"Same parameters should produce same hash: {hash1a} != {hash1b}"


def test_slpccd_different_split_different_hash(create_dummy_slpccd_files):
    """Test that different splits produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_slpccd_files(temp_dir)
        
        dataset_train = SLPCCDDataset(data_root=temp_dir, split='train')
        dataset_val = SLPCCDDataset(data_root=temp_dir, split='val')
        dataset_test = SLPCCDDataset(data_root=temp_dir, split='test')
        
        hash_train = dataset_train.get_cache_version_hash()
        hash_val = dataset_val.get_cache_version_hash()
        hash_test = dataset_test.get_cache_version_hash()
        
        assert hash_train != hash_val, f"Different splits should produce different hashes: {hash_train} == {hash_val}"
        assert hash_train != hash_test, f"Different splits should produce different hashes: {hash_train} == {hash_test}"
        assert hash_val != hash_test, f"Different splits should produce different hashes: {hash_val} == {hash_test}"


def test_slpccd_different_num_points_different_hash(create_dummy_slpccd_files):
    """Test that different num_points values produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_slpccd_files(temp_dir)
        
        dataset_4096 = SLPCCDDataset(data_root=temp_dir, split='train', num_points=4096)
        dataset_8192 = SLPCCDDataset(data_root=temp_dir, split='train', num_points=8192)
        
        hash_4096 = dataset_4096.get_cache_version_hash()
        hash_8192 = dataset_8192.get_cache_version_hash()
        
        assert hash_4096 != hash_8192, f"Different num_points should produce different hashes: {hash_4096} == {hash_8192}"


def test_slpccd_different_random_subsample_different_hash(create_dummy_slpccd_files):
    """Test that different random_subsample values produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_slpccd_files(temp_dir)
        
        dataset_random = SLPCCDDataset(data_root=temp_dir, split='train', random_subsample=True)
        dataset_no_random = SLPCCDDataset(data_root=temp_dir, split='train', random_subsample=False)
        
        hash_random = dataset_random.get_cache_version_hash()
        hash_no_random = dataset_no_random.get_cache_version_hash()
        
        assert hash_random != hash_no_random, f"Different random_subsample should produce different hashes: {hash_random} == {hash_no_random}"


def test_slpccd_different_use_hierarchy_different_hash(create_dummy_slpccd_files):
    """Test that different use_hierarchy values produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_slpccd_files(temp_dir)
        
        dataset_hierarchy = SLPCCDDataset(data_root=temp_dir, split='train', use_hierarchy=True)
        dataset_no_hierarchy = SLPCCDDataset(data_root=temp_dir, split='train', use_hierarchy=False)
        
        hash_hierarchy = dataset_hierarchy.get_cache_version_hash()
        hash_no_hierarchy = dataset_no_hierarchy.get_cache_version_hash()
        
        assert hash_hierarchy != hash_no_hierarchy, f"Different use_hierarchy should produce different hashes: {hash_hierarchy} == {hash_no_hierarchy}"


def test_slpccd_different_hierarchy_levels_different_hash(create_dummy_slpccd_files):
    """Test that different hierarchy_levels values produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_slpccd_files(temp_dir)
        
        dataset_2_levels = SLPCCDDataset(data_root=temp_dir, split='train', hierarchy_levels=2)
        dataset_3_levels = SLPCCDDataset(data_root=temp_dir, split='train', hierarchy_levels=3)
        
        hash_2_levels = dataset_2_levels.get_cache_version_hash()
        hash_3_levels = dataset_3_levels.get_cache_version_hash()
        
        assert hash_2_levels != hash_3_levels, f"Different hierarchy_levels should produce different hashes: {hash_2_levels} == {hash_3_levels}"


def test_slpccd_different_knn_size_different_hash(create_dummy_slpccd_files):
    """Test that different knn_size values produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_slpccd_files(temp_dir)
        
        dataset_knn_8 = SLPCCDDataset(data_root=temp_dir, split='train', knn_size=8)
        dataset_knn_16 = SLPCCDDataset(data_root=temp_dir, split='train', knn_size=16)
        
        hash_knn_8 = dataset_knn_8.get_cache_version_hash()
        hash_knn_16 = dataset_knn_16.get_cache_version_hash()
        
        assert hash_knn_8 != hash_knn_16, f"Different knn_size should produce different hashes: {hash_knn_8} == {hash_knn_16}"


def test_slpccd_different_cross_knn_size_different_hash(create_dummy_slpccd_files):
    """Test that different cross_knn_size values produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_slpccd_files(temp_dir)
        
        dataset_cross_8 = SLPCCDDataset(data_root=temp_dir, split='train', cross_knn_size=8)
        dataset_cross_16 = SLPCCDDataset(data_root=temp_dir, split='train', cross_knn_size=16)
        
        hash_cross_8 = dataset_cross_8.get_cache_version_hash()
        hash_cross_16 = dataset_cross_16.get_cache_version_hash()
        
        assert hash_cross_8 != hash_cross_16, f"Different cross_knn_size should produce different hashes: {hash_cross_8} == {hash_cross_16}"


def test_slpccd_different_data_root_different_hash(create_dummy_slpccd_files):
    """Test that different data roots produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_slpccd_files(temp_dir1)
            create_dummy_slpccd_files(temp_dir2)
            
            dataset1 = SLPCCDDataset(data_root=temp_dir1, split='train')
            dataset2 = SLPCCDDataset(data_root=temp_dir2, split='train')
            
            hash1 = dataset1.get_cache_version_hash()
            hash2 = dataset2.get_cache_version_hash()
            
            assert hash1 != hash2, f"Different data roots should produce different hashes: {hash1} == {hash2}"


def test_slpccd_hash_format(create_dummy_slpccd_files):
    """Test that hash is in correct format."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_slpccd_files(temp_dir)
        
        dataset = SLPCCDDataset(data_root=temp_dir, split='train')
        hash_val = dataset.get_cache_version_hash()
        
        # Should be a string
        assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"
        
        # Should be 16 characters (xxhash format)
        assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"
        
        # Should be hexadecimal
        assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"


def test_slpccd_comprehensive_no_hash_collisions(create_dummy_slpccd_files):
    """Test that different configurations produce unique hashes (no collisions)."""
    
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_slpccd_files(temp_dir1)
            create_dummy_slpccd_files(temp_dir2)
            
            # Test various parameter combinations
            configs = [
                {'data_root': temp_dir1, 'split': 'train', 'num_points': 4096, 'random_subsample': True},
                {'data_root': temp_dir1, 'split': 'train', 'num_points': 8192, 'random_subsample': True},
                {'data_root': temp_dir1, 'split': 'train', 'num_points': 4096, 'random_subsample': False},
                {'data_root': temp_dir1, 'split': 'val', 'num_points': 4096, 'random_subsample': True},
                {'data_root': temp_dir1, 'split': 'test', 'num_points': 4096, 'random_subsample': True},
                {'data_root': temp_dir1, 'split': 'train', 'use_hierarchy': False},
                {'data_root': temp_dir1, 'split': 'train', 'hierarchy_levels': 2},
                {'data_root': temp_dir1, 'split': 'train', 'knn_size': 8},
                {'data_root': temp_dir1, 'split': 'train', 'cross_knn_size': 8},
                {'data_root': temp_dir2, 'split': 'train', 'num_points': 4096, 'random_subsample': True},
            ]
            
            hashes = []
            for config in configs:
                dataset = SLPCCDDataset(**config)
                hash_val = dataset.get_cache_version_hash()
                
                # Check for collision
                assert hash_val not in hashes, f"Hash collision detected for config {config}: hash {hash_val} already exists"
                hashes.append(hash_val)
            
            # Verify we generated the expected number of unique hashes
            assert len(hashes) == len(configs), f"Expected {len(configs)} unique hashes, got {len(hashes)}"