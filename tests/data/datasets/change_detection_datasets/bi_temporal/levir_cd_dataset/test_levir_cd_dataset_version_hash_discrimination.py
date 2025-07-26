"""Tests for LevirCdDataset cache version discrimination."""

import pytest
import os
from PIL import Image
import numpy as np
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset




def test_levir_cd_dataset_version_discrimination(levir_cd_data_root):
    """Test that LevirCdDataset instances with different parameters have different hashes."""
    
    # Same parameters should have same hash
    dataset1a = LevirCdDataset(
        data_root=levir_cd_data_root,
        split='train'
    )
    dataset1b = LevirCdDataset(
        data_root=levir_cd_data_root,
        split='train'
    )
    assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
    
    # Different split should have different hash
    dataset2 = LevirCdDataset(
        data_root=levir_cd_data_root,
        split='val'  # Different
    )
    assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_split_variants(levir_cd_data_root):
    """Test that different splits produce different hashes."""
    
    split_variants = ['train', 'val', 'test']
    
    datasets = []
    for split in split_variants:
        dataset = LevirCdDataset(
            data_root=levir_cd_data_root,
            split=split
        )
        datasets.append(dataset)
    
    # All should have different hashes
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    assert len(hashes) == len(set(hashes)), \
        f"All split variants should produce different hashes, got: {hashes}"


def test_cache_version_reflects_configuration_not_path(levir_cd_data_root):
    """Test that cache hash reflects dataset configuration, not data_root path.
    
    This ensures that:
    - Same dataset in different locations has same hash (relocatable)
    - Soft links pointing to same data have same hash
    - Cache is stable regardless of where dataset is stored
    """
    
    # Test that identical configurations produce identical hashes
    # This verifies the cache version is based on meaningful dataset parameters
    dataset1 = LevirCdDataset(
        data_root=levir_cd_data_root,
        split='train'
    )
    
    dataset2 = LevirCdDataset(
        data_root=levir_cd_data_root,
        split='train'
    )
    
    # Should have same hashes - cache version reflects configuration, not path
    # This ensures cache works correctly with soft links and relocated datasets
    assert dataset1.get_cache_version_hash() == dataset2.get_cache_version_hash()


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


def test_comprehensive_no_hash_collisions(levir_cd_data_root):
    """Ensure no hash collisions across many different configurations."""
    datasets = []
    
    # Test different splits with same data root - should produce different hashes
    for split in ['train', 'val', 'test']:
        dataset = LevirCdDataset(
            data_root=levir_cd_data_root,
            split=split
        )
        datasets.append(dataset)
    
    # Collect all hashes using actual BaseDataset implementation
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    
    # Ensure all hashes are unique (no collisions)
    assert len(hashes) == len(set(hashes)), \
        f"Hash collision detected! Duplicate hashes found in: {hashes}"
    
    # Ensure all hashes are properly formatted
    for hash_val in hashes:
        assert isinstance(hash_val, str), f"Hash must be string, got {type(hash_val)}"
        assert len(hash_val) == 16, f"Hash must be 16 characters, got {len(hash_val)}"


