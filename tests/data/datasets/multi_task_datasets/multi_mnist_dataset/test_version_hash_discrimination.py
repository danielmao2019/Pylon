"""Tests for MultiMNISTDataset cache version discrimination."""

import pytest
import tempfile
import os
from data.datasets.multi_task_datasets.multi_mnist_dataset import MultiMNISTDataset


def test_multi_mnist_dataset_version_discrimination():
    """Test that MultiMNISTDataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Same parameters should have same hash
        dataset1a = MultiMNISTDataset(
            data_root=temp_dir,
            split='train'
        )
        dataset1b = MultiMNISTDataset(
            data_root=temp_dir,
            split='train'
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different split should have different hash
        dataset2 = MultiMNISTDataset(
            data_root=temp_dir,
            split='val'  # Different
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different data_root should have different hash
        with tempfile.TemporaryDirectory() as temp_dir2:
            dataset3 = MultiMNISTDataset(
                data_root=temp_dir2,  # Different
                split='train'
            )
            assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()


def test_split_variants():
    """Test that different splits produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        split_variants = ['train', 'val']
        
        datasets = []
        for split in split_variants:
            dataset = MultiMNISTDataset(
                data_root=temp_dir,
                split=split
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All split variants should produce different hashes, got: {hashes}"


def test_inherited_parameters_affect_version_hash():
    """Test that parameters inherited from BaseDataset affect version hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_args = {
            'data_root': temp_dir,
            'split': 'train',
        }
        
        # Test inherited parameters from BaseDataset
        parameter_variants = [
            ('initial_seed', 42),  # Different from default None
            ('cache_size', 1000),  # Different from default
        ]
        
        dataset1 = MultiMNISTDataset(**base_args)
        
        for param_name, new_value in parameter_variants:
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = MultiMNISTDataset(**modified_args)
            
            assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
                f"Inherited parameter {param_name} should affect cache version hash"


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        datasets = []
        
        # Generate different dataset configurations
        for split in ['train', 'val']:
            for initial_seed in [None, 42, 123]:
                datasets.append(MultiMNISTDataset(
                    data_root=temp_dir,
                    split=split,
                    initial_seed=initial_seed
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


