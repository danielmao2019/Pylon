"""Tests for KITTIDataset cache version discrimination."""

import pytest
import copy
from utils.builders.builder import build_from_config


@pytest.mark.parametrize('kitti_dataset_config', ['train'], indirect=True)
def test_kitti_dataset_version_discrimination(kitti_dataset_config):
    """Test that KITTIDataset instances with different parameters have different hashes."""
    # Same parameters should have same hash
    config1a = copy.deepcopy(kitti_dataset_config)
    config1b = copy.deepcopy(kitti_dataset_config)
    
    dataset1a = build_from_config(config1a)
    dataset1b = build_from_config(config1b)
    assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
    
    # Different split should have different hash
    config2 = copy.deepcopy(kitti_dataset_config)
    config2['args']['split'] = 'val'  # Different
    dataset2 = build_from_config(config2)
    assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()


@pytest.mark.parametrize('kitti_dataset_config', ['train'], indirect=True)
def test_split_variants(kitti_dataset_config):
    """Test that different splits produce different hashes."""
    split_variants = ['train', 'val', 'test']
    
    datasets = []
    for split in split_variants:
        config = copy.deepcopy(kitti_dataset_config)
        config['args']['split'] = split
        dataset = build_from_config(config)
        datasets.append(dataset)
    
    # All should have different hashes
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    assert len(hashes) == len(set(hashes)), \
        f"All split variants should produce different hashes, got: {hashes}"


@pytest.mark.parametrize('kitti_dataset_config', ['train'], indirect=True)
def test_inherited_parameters_affect_version_hash(kitti_dataset_config):
    """Test that parameters inherited from BaseDataset affect version hash."""
    # Test inherited parameters from BaseDataset
    parameter_variants = [
        ('base_seed', 42),  # Different from default 0
    ]
    
    dataset1 = build_from_config(kitti_dataset_config)
    
    for param_name, new_value in parameter_variants:
        modified_config = copy.deepcopy(kitti_dataset_config)
        modified_config['args'][param_name] = new_value
        dataset2 = build_from_config(modified_config)
        
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
            f"Inherited parameter {param_name} should affect cache version hash"


@pytest.mark.parametrize('kitti_dataset_config', ['train'], indirect=True)
def test_comprehensive_no_hash_collisions(kitti_dataset_config):
    """Ensure no hash collisions across many different configurations."""
    datasets = []
    
    # Generate different dataset configurations
    for split in ['train', 'val', 'test']:
        for base_seed_val in [None, 42, 123]:
            config = copy.deepcopy(kitti_dataset_config)
            config['args']['split'] = split
            if base_seed_val is not None:
                config['args']['base_seed'] = base_seed_val
            
            dataset = build_from_config(config)
            datasets.append(dataset)
    
    # Collect all hashes
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    
    # Ensure all hashes are unique (no collisions)
    assert len(hashes) == len(set(hashes)), \
        f"Hash collision detected! Duplicate hashes found in: {hashes}"
    
    # Ensure all hashes are properly formatted
    for hash_val in hashes:
        assert isinstance(hash_val, str), f"Hash must be string, got {type(hash_val)}"
        assert len(hash_val) == 16, f"Hash must be 16 characters, got {len(hash_val)}"
