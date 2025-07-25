"""Tests for LevirCdDataset cache version discrimination."""

import pytest
import tempfile
import os
from PIL import Image
import numpy as np
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset




def test_levir_cd_dataset_version_discrimination(create_dummy_levir_structure):
    """Test that LevirCdDataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_levir_structure(temp_dir)
        
        # Same parameters should have same hash
        dataset1a = LevirCdDataset(
            data_root=temp_dir,
            split='train'
        )
        dataset1b = LevirCdDataset(
            data_root=temp_dir,
            split='train'
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different split should have different hash
        dataset2 = LevirCdDataset(
            data_root=temp_dir,
            split='val'  # Different
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different data_root should have different hash
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_levir_structure(temp_dir2)
            dataset3 = LevirCdDataset(
                data_root=temp_dir2,  # Different
                split='train'
            )
            assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()


def test_split_variants(create_dummy_levir_structure):
    """Test that different splits produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_levir_structure(temp_dir)
        
        split_variants = ['train', 'val', 'test']
        
        datasets = []
        for split in split_variants:
            dataset = LevirCdDataset(
                data_root=temp_dir,
                split=split
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All split variants should produce different hashes, got: {hashes}"


def test_different_file_structures(create_dummy_levir_structure):
    """Test that different file structures produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create first structure
        create_dummy_levir_structure(temp_dir)
        
        dataset1 = LevirCdDataset(
            data_root=temp_dir,
            split='train'
        )
        
        # Add an additional file to the structure
        split_dir = os.path.join(temp_dir, 'train')
        filename = 'test_9999.png'
        
        img_a = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img_b = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        label = Image.fromarray(np.random.randint(0, 2, (256, 256), dtype=np.uint8))
        
        img_a.save(os.path.join(split_dir, 'A', filename))
        img_b.save(os.path.join(split_dir, 'B', filename))
        label.save(os.path.join(split_dir, 'label', filename))
        
        dataset2 = LevirCdDataset(
            data_root=temp_dir,
            split='train'
        )
        
        # Should have different hashes due to different file count
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_inherited_parameters_affect_version_hash(create_dummy_levir_structure):
    """Test that parameters inherited from BaseDataset affect version hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_levir_structure(temp_dir)
        
        base_args = {
            'data_root': temp_dir,
            'split': 'train',
        }
        
        # Test inherited parameters from BaseDataset
        parameter_variants = [
            ('initial_seed', 42),  # Different from default None
            ('cache_size', 1000),  # Different from default
        ]
        
        dataset1 = LevirCdDataset(**base_args)
        
        for param_name, new_value in parameter_variants:
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = LevirCdDataset(**modified_args)
            
            assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
                f"Inherited parameter {param_name} should affect cache version hash"


def test_comprehensive_no_hash_collisions(create_dummy_levir_structure):
    """Ensure no hash collisions across many different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_levir_structure(temp_dir)
        
        datasets = []
        
        # Generate different dataset configurations
        for split in ['train', 'val', 'test']:
            for initial_seed in [None, 42, 123]:
                datasets.append(LevirCdDataset(
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


