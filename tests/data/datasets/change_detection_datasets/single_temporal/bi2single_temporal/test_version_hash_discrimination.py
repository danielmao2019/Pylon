"""Test version hash discrimination for Bi2SingleTemporal.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
import tempfile
import os
from data.datasets.change_detection_datasets.single_temporal.bi2single_temporal_dataset import Bi2SingleTemporal
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset



def test_bi2single_temporal_same_parameters_same_hash(create_dummy_levir_cd_files):
    """Test that identical parameters produce identical hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_levir_cd_files(temp_dir)
        
        # Create source datasets
        source1a = LevirCdDataset(data_root=temp_dir, split='train')
        source1b = LevirCdDataset(data_root=temp_dir, split='train')
        
        # Same parameters should produce same hash
        dataset1a = Bi2SingleTemporal(source=source1a)
        dataset1b = Bi2SingleTemporal(source=source1b)
        
        hash1a = dataset1a.get_cache_version_hash()
        hash1b = dataset1b.get_cache_version_hash()
        
        assert hash1a == hash1b, f"Same parameters should produce same hash: {hash1a} != {hash1b}"


def test_bi2single_temporal_different_source_different_hash(create_dummy_levir_cd_files):
    """Test that different source datasets produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_levir_cd_files(temp_dir)
        
        # Create different source datasets
        source_train = LevirCdDataset(data_root=temp_dir, split='train')
        source_test = LevirCdDataset(data_root=temp_dir, split='test')
        
        dataset_train = Bi2SingleTemporal(source=source_train)
        dataset_test = Bi2SingleTemporal(source=source_test)
        
        hash_train = dataset_train.get_cache_version_hash()
        hash_test = dataset_test.get_cache_version_hash()
        
        assert hash_train != hash_test, f"Different source datasets should produce different hashes: {hash_train} == {hash_test}"


def test_bi2single_temporal_different_source_data_root_different_hash(create_dummy_levir_cd_files):
    """Test that different source data roots produce different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_levir_cd_files(temp_dir1)
            create_dummy_levir_cd_files(temp_dir2)
            
            source1 = LevirCdDataset(data_root=temp_dir1, split='train')
            source2 = LevirCdDataset(data_root=temp_dir2, split='train')
            
            dataset1 = Bi2SingleTemporal(source=source1)
            dataset2 = Bi2SingleTemporal(source=source2)
            
            hash1 = dataset1.get_cache_version_hash()
            hash2 = dataset2.get_cache_version_hash()
            
            assert hash1 != hash2, f"Different source data roots should produce different hashes: {hash1} == {hash2}"


def test_bi2single_temporal_hash_format(create_dummy_levir_cd_files):
    """Test that hash is in correct format."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_levir_cd_files(temp_dir)
        
        source = LevirCdDataset(data_root=temp_dir, split='train')
        dataset = Bi2SingleTemporal(source=source)
        hash_val = dataset.get_cache_version_hash()
        
        # Should be a string
        assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"
        
        # Should be 16 characters (xxhash format)
        assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"
        
        # Should be hexadecimal
        assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"


def test_bi2single_temporal_comprehensive_no_hash_collisions(create_dummy_levir_cd_files):
    """Test that different configurations produce unique hashes (no collisions)."""
    
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_levir_cd_files(temp_dir1)
            create_dummy_levir_cd_files(temp_dir2)
            
            # Create different source datasets
            sources = [
                LevirCdDataset(data_root=temp_dir1, split='train'),
                LevirCdDataset(data_root=temp_dir1, split='test'),
                LevirCdDataset(data_root=temp_dir2, split='train'),
                LevirCdDataset(data_root=temp_dir2, split='test'),
            ]
            
            hashes = []
            for source in sources:
                dataset = Bi2SingleTemporal(source=source)
                hash_val = dataset.get_cache_version_hash()
                
                # Check for collision
                assert hash_val not in hashes, f"Hash collision detected for source {source}: hash {hash_val} already exists"
                hashes.append(hash_val)
            
            # Verify we generated the expected number of unique hashes
            assert len(hashes) == len(sources), f"Expected {len(sources)} unique hashes, got {len(hashes)}"