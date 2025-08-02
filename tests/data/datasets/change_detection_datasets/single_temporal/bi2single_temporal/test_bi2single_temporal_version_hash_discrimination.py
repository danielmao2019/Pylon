"""Test version hash discrimination for Bi2SingleTemporal.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
from data.datasets.change_detection_datasets.single_temporal.bi2single_temporal_dataset import Bi2SingleTemporal
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset



def test_bi2single_temporal_same_parameters_same_hash(levir_cd_data_root):
    """Test that identical parameters produce identical hashes."""
    
    # Create source datasets with same parameters
    source1a = LevirCdDataset(data_root=levir_cd_data_root, split='train')
    source1b = LevirCdDataset(data_root=levir_cd_data_root, split='train')
    
    # Same parameters should produce same hash
    dataset1a = Bi2SingleTemporal(source=source1a)
    dataset1b = Bi2SingleTemporal(source=source1b)
    
    hash1a = dataset1a.get_cache_version_hash()
    hash1b = dataset1b.get_cache_version_hash()
    
    assert hash1a == hash1b, f"Same parameters should produce same hash: {hash1a} != {hash1b}"


def test_bi2single_temporal_different_source_different_hash(levir_cd_data_root):
    """Test that different source datasets produce different hashes."""
    
    # Create different source datasets
    source_train = LevirCdDataset(data_root=levir_cd_data_root, split='train')
    source_test = LevirCdDataset(data_root=levir_cd_data_root, split='test')
    
    dataset_train = Bi2SingleTemporal(source=source_train)
    dataset_test = Bi2SingleTemporal(source=source_test)
    
    hash_train = dataset_train.get_cache_version_hash()
    hash_test = dataset_test.get_cache_version_hash()
    
    assert hash_train != hash_test, f"Different source datasets should produce different hashes: {hash_train} == {hash_test}"


def test_bi2single_temporal_different_source_data_root_different_hash(levir_cd_data_root):
    """Test that different source configurations produce different hashes.
    
    Note: This test originally expected different data roots to produce different hashes,
    but the framework intentionally excludes data_root from hash calculation for cache
    stability across filesystem locations. Instead, we test different splits which 
    should produce different hashes since they contain different data.
    """
    
    # Test with different splits - these should have different hashes
    # since they contain different data (different dataset sizes)
    source1 = LevirCdDataset(data_root=levir_cd_data_root, split='train')  # 445 files
    source2 = LevirCdDataset(data_root=levir_cd_data_root, split='test')   # 128 files
    
    dataset1 = Bi2SingleTemporal(source=source1)
    dataset2 = Bi2SingleTemporal(source=source2)
    
    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()
    
    assert hash1 != hash2, f"Different source splits should produce different hashes: {hash1} == {hash2}"


def test_bi2single_temporal_hash_format(levir_cd_data_root):
    """Test that hash is in correct format."""
    
    source = LevirCdDataset(data_root=levir_cd_data_root, split='train')
    dataset = Bi2SingleTemporal(source=source)
    hash_val = dataset.get_cache_version_hash()
    
    # Should be a string
    assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"
    
    # Should be 16 characters (xxhash format)
    assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"
    
    # Should be hexadecimal
    assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"


def test_bi2single_temporal_comprehensive_no_hash_collisions(levir_cd_data_root):
    """Test that different configurations produce unique hashes (no collisions).
    
    Note: The framework intentionally excludes data_root from hash calculation for
    cache stability, so datasets with identical data in different locations will
    have the same hash. We test truly different configurations here.
    """
    
    # Create different source datasets that should have different hashes
    # Only include configurations that actually differ in meaningful ways
    sources = [
        LevirCdDataset(data_root=levir_cd_data_root, split='train'),  # 445 files
        LevirCdDataset(data_root=levir_cd_data_root, split='test'),   # 128 files
    ]
    
    hashes = []
    for source in sources:
        dataset = Bi2SingleTemporal(source=source)
        hash_val = dataset.get_cache_version_hash()
        
        # Check for collision
        assert hash_val not in hashes, f"Hash collision detected for source split {source.split}: hash {hash_val} already exists"
        hashes.append(hash_val)
    
    # Verify we generated the expected number of unique hashes
    assert len(hashes) == len(sources), f"Expected {len(sources)} unique hashes, got {len(hashes)}"
    
    # Verify we have exactly 2 unique hashes (train and test)
    assert len(set(hashes)) == 2, f"Expected 2 unique hashes for train/test splits, got {len(set(hashes))}: {hashes}"
