"""Tests for PASCALContextDataset cache version discrimination."""

import pytest
from data.datasets.multi_task_datasets.pascal_context_dataset import PASCALContextDataset


def test_pascal_context_dataset_version_discrimination(pascal_context_data_root):
    """Test that PASCALContextDataset instances with different parameters have different hashes."""
    # Same parameters should have same hash
    dataset1a = PASCALContextDataset(
        data_root=pascal_context_data_root,
        split='train',
        num_human_parts=6,
        area_thres=0
    )
    dataset1b = PASCALContextDataset(
        data_root=pascal_context_data_root,
        split='train',
        num_human_parts=6,
        area_thres=0
    )
    assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
    
    # Different split should have different hash
    dataset2 = PASCALContextDataset(
        data_root=pascal_context_data_root,
        split='val',  # Different
        num_human_parts=6,
        area_thres=0
    )
    assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
    
    # Different num_human_parts should have different hash
    dataset3 = PASCALContextDataset(
        data_root=pascal_context_data_root,
        split='train',
        num_human_parts=14,  # Different (valid values: 1, 4, 6, 14)
        area_thres=0
    )
    assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()
    
    # Different area_thres should have different hash
    dataset4 = PASCALContextDataset(
        data_root=pascal_context_data_root,
        split='train',
        num_human_parts=6,
        area_thres=100  # Different
    )
    assert dataset1a.get_cache_version_hash() != dataset4.get_cache_version_hash()


def test_split_variants(pascal_context_data_root):
    """Test that different splits produce different hashes."""
    split_variants = ['train', 'val']
    
    datasets = []
    for split in split_variants:
        dataset = PASCALContextDataset(
            data_root=pascal_context_data_root,
            split=split,
            num_human_parts=6,
            area_thres=0
        )
        datasets.append(dataset)
    
    # All should have different hashes
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    assert len(hashes) == len(set(hashes)), \
        f"All split variants should produce different hashes, got: {hashes}"


def test_num_human_parts_variants(pascal_context_data_root):
    """Test that different num_human_parts produce different hashes."""
    parts_variants = [4, 6, 14]  # Valid values from HUMAN_PART dictionary
    
    datasets = []
    for num_parts in parts_variants:
        dataset = PASCALContextDataset(
            data_root=pascal_context_data_root,
            split='train',
            num_human_parts=num_parts,
            area_thres=0
        )
        datasets.append(dataset)
    
    # All should have different hashes
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    assert len(hashes) == len(set(hashes)), \
        f"All num_human_parts variants should produce different hashes, got: {hashes}"


def test_area_threshold_variants(pascal_context_data_root):
    """Test that different area thresholds produce different hashes."""
    area_variants = [0, 50, 100]
    
    datasets = []
    for area_thres in area_variants:
        dataset = PASCALContextDataset(
            data_root=pascal_context_data_root,
            split='train',
            num_human_parts=6,
            area_thres=area_thres
        )
        datasets.append(dataset)
    
    # All should have different hashes
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    assert len(hashes) == len(set(hashes)), \
        f"All area_thres variants should produce different hashes, got: {hashes}"


def test_inherited_parameters_affect_version_hash(pascal_context_data_root):
    """Test that parameters inherited from BaseDataset affect version hash."""
    base_args = {
        'data_root': pascal_context_data_root,
        'split': 'train',
        'num_human_parts': 6,
        'area_thres': 0,
    }
    
    # Test inherited parameters from BaseDataset
    parameter_variants = [
        ('base_seed', 42),  # Different from default None
    ]
    
    dataset1 = PASCALContextDataset(**base_args)
    
    for param_name, new_value in parameter_variants:
        modified_args = base_args.copy()
        modified_args[param_name] = new_value
        dataset2 = PASCALContextDataset(**modified_args)
        
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
            f"Inherited parameter {param_name} should affect cache version hash"


def test_comprehensive_no_hash_collisions(pascal_context_data_root):
    """Ensure no hash collisions across many different configurations."""
    datasets = []
    
    # Generate different dataset configurations
    for split in ['train', 'val']:
        for num_parts in [6, 14]:  # Valid values from HUMAN_PART dictionary
            for area_thres in [0, 50]:
                for base_seed_val in [None, 42, 123]:
                    datasets.append(PASCALContextDataset(
                        data_root=pascal_context_data_root,
                        split=split,
                        num_human_parts=num_parts,
                        area_thres=area_thres,
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
