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


def test_cache_stability_with_file_modifications(create_dummy_levir_structure):
    """Test that cache hash remains stable when file contents are modified (correct caching behavior)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create first structure
        create_dummy_levir_structure(temp_dir)
        
        dataset1 = LevirCdDataset(
            data_root=temp_dir,
            split='train'
        )
        
        # Modify content of existing files (without changing count)
        split_dir = os.path.join(temp_dir, 'train')
        filename = 'test_0000.png'  # Modify existing file
        
        # Overwrite existing files with different content
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
        
        # Should have same hashes - cache version reflects configuration, not file content
        # This ensures cache stability when data file contents are modified
        assert dataset1.get_cache_version_hash() == dataset2.get_cache_version_hash()


def test_parameters_that_dont_affect_version_hash(create_dummy_levir_structure):
    """Test that cache/processing parameters don't affect version hash (correct caching behavior)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_levir_structure(temp_dir)
        
        base_args = {
            'data_root': temp_dir,
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


def test_comprehensive_no_hash_collisions(create_dummy_levir_structure):
    """Ensure no hash collisions across many different configurations."""
    datasets = []
    temp_dirs = []
    
    try:
        # Generate different dataset configurations using different data_root paths
        for _ in range(3):
            temp_dir = tempfile.mkdtemp()
            temp_dirs.append(temp_dir)
            create_dummy_levir_structure(temp_dir)
            
            # Test different splits - use val and test which have smaller sizes
            for split in ['val', 'test']:
                dataset = LevirCdDataset(
                    data_root=temp_dir,
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
    
    finally:
        # Clean up temporary directories
        import shutil
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)


