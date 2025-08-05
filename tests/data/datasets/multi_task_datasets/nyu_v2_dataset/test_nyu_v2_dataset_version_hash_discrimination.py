"""Tests for NYUv2Dataset cache version discrimination."""

import pytest
from data.datasets.multi_task_datasets.nyu_v2_dataset import NYUv2Dataset


def test_nyuv2_dataset_version_discrimination(nyu_v2_data_root):
    """Test that NYUv2Dataset instances with different parameters have different hashes."""
    # Same parameters should have same hash
    dataset1a = NYUv2Dataset(
        data_root=nyu_v2_data_root,
        split='train',
        semantic_granularity='coarse'
    )
    dataset1b = NYUv2Dataset(
        data_root=nyu_v2_data_root,
        split='train',
        semantic_granularity='coarse'
    )
    assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
    
    # Different split should have different hash
    dataset2 = NYUv2Dataset(
        data_root=nyu_v2_data_root,
        split='val',  # Different
        semantic_granularity='coarse'
    )
    assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
    
    # Different semantic_granularity should have different hash
    dataset3 = NYUv2Dataset(
        data_root=nyu_v2_data_root,
        split='train',
        semantic_granularity='fine'  # Different
    )
    assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()


def test_split_variants(nyu_v2_data_root):
    """Test that different splits produce different hashes."""
    split_variants = ['train', 'val']
    
    datasets = []
    for split in split_variants:
        dataset = NYUv2Dataset(
            data_root=nyu_v2_data_root,
            split=split,
            semantic_granularity='coarse'
        )
        datasets.append(dataset)
    
    # All should have different hashes
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    assert len(hashes) == len(set(hashes)), \
        f"All split variants should produce different hashes, got: {hashes}"


def test_semantic_granularity_variants(nyu_v2_data_root):
    """Test that different semantic granularities produce different hashes."""
    granularity_variants = ['fine', 'coarse']
    
    datasets = []
    for granularity in granularity_variants:
        dataset = NYUv2Dataset(
            data_root=nyu_v2_data_root,
            split='train',
            semantic_granularity=granularity
        )
        datasets.append(dataset)
    
    # All should have different hashes
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    assert len(hashes) == len(set(hashes)), \
        f"All granularity variants should produce different hashes, got: {hashes}"


def test_inherited_parameters_affect_version_hash(nyu_v2_data_root):
    """Test that parameters inherited from BaseDataset affect version hash."""
    base_args = {
        'data_root': nyu_v2_data_root,
        'split': 'train',
        'semantic_granularity': 'coarse',
    }
    
    # Test inherited parameters from BaseDataset
    parameter_variants = [
        ('base_seed', 42),  # Different from default None
    ]
    
    dataset1 = NYUv2Dataset(**base_args)
    
    for param_name, new_value in parameter_variants:
        modified_args = base_args.copy()
        modified_args[param_name] = new_value
        dataset2 = NYUv2Dataset(**modified_args)
        
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
            f"Inherited parameter {param_name} should affect cache version hash"


def test_comprehensive_no_hash_collisions(nyu_v2_data_root):
    """Ensure no hash collisions across many different configurations."""
    datasets = []
    
    # Generate different dataset configurations
    for split in ['train', 'val']:
        for granularity in ['fine', 'coarse']:
            for base_seed_val in [None, 42, 123]:
                datasets.append(NYUv2Dataset(
                    data_root=nyu_v2_data_root,
                    split=split,
                    semantic_granularity=granularity,
                    base_seed=base_seed_val
                ))
    
    # Collect all hashes
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    
    # Ensure all hashes are unique (no collisions)
    assert len(hashes) == len(set(hashes)), \
        f"Hash collision detected! Duplicate hashes found in: {hashes}"
    
    # Ensure all hashes are properly formatted
    for hash_val in hashes:
        assert isinstance(hash_val, str), f"Hash must be string, got {type(hash_val)}"
        assert len(hash_val) == 16, f"Hash must be 16 characters, got {len(hash_val)}"
