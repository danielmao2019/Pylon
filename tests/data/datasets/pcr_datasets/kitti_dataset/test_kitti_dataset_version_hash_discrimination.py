"""Tests for KITTIDataset cache version discrimination."""

import pytest
import tempfile
import os
import json
import numpy as np
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset




def test_kitti_dataset_version_discrimination(create_dummy_kitti_structure):
    """Test that KITTIDataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kitti_structure(temp_dir)
        
        # Same parameters should have same hash
        dataset1a = KITTIDataset(
            data_root=temp_dir,
            split='train'
        )
        dataset1b = KITTIDataset(
            data_root=temp_dir,
            split='train'
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different split should have different hash
        dataset2 = KITTIDataset(
            data_root=temp_dir,
            split='val'  # Different
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different data_root should have SAME hash (data_root excluded from versioning)
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_kitti_structure(temp_dir2)
            dataset3 = KITTIDataset(
                data_root=temp_dir2,  # Different
                split='train'
            )
            assert dataset1a.get_cache_version_hash() == dataset3.get_cache_version_hash()


def test_split_variants(create_dummy_kitti_structure):
    """Test that different splits produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kitti_structure(temp_dir)
        
        split_variants = ['train', 'val', 'test']
        
        datasets = []
        for split in split_variants:
            dataset = KITTIDataset(
                data_root=temp_dir,
                split=split
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All split variants should produce different hashes, got: {hashes}"


def test_different_sequence_data(create_dummy_kitti_structure):
    """Test that different sequence data produces different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create first structure
        create_dummy_kitti_structure(temp_dir)
        
        dataset1 = KITTIDataset(
            data_root=temp_dir,
            split='train'
        )
        
        # Modify the sequence data (add more points to sequence 00)
        seq_dir = os.path.join(temp_dir, 'sequences', '00', 'velodyne')
        bin_file = os.path.join(seq_dir, '000000.bin')
        
        # Add more points to the first file
        additional_data = np.array([
            [10.0, 10.0, 10.0, 0.8],
            [11.0, 10.0, 10.0, 0.8],
        ], dtype=np.float32)
        
        # Read existing data and append
        existing_data = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        modified_data = np.vstack([existing_data, additional_data])
        modified_data.tofile(bin_file)
        
        dataset2 = KITTIDataset(
            data_root=temp_dir,
            split='train'
        )
        
        # Should have different hashes due to different data
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_different_pose_data(create_dummy_kitti_structure):
    """Test that different pose data produces different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kitti_structure(temp_dir)
        
        dataset1 = KITTIDataset(
            data_root=temp_dir,
            split='train'
        )
        
        # Modify pose file
        pose_file = os.path.join(temp_dir, 'poses', '00.txt')
        with open(pose_file, 'w') as f:
            # Write modified poses (different translations)
            for i in range(5):
                pose = [1.0, 0.0, 0.0, i*20.0,  # Different translation values
                       0.0, 1.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0]
                f.write(' '.join(map(str, pose)) + '\n')
        
        dataset2 = KITTIDataset(
            data_root=temp_dir,
            split='train'
        )
        
        # Should have different hashes due to different pose data
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_inherited_parameters_affect_version_hash(create_dummy_kitti_structure):
    """Test that parameters inherited from BaseDataset affect version hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kitti_structure(temp_dir)
        
        base_args = {
            'data_root': temp_dir,
            'split': 'train',
        }
        
        # Test inherited parameters from BaseDataset
        parameter_variants = [
            ('base_seed', 42),  # Different from default 0
        ]
        
        dataset1 = KITTIDataset(**base_args)
        
        for param_name, new_value in parameter_variants:
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = KITTIDataset(**modified_args)
            
            assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
                f"Inherited parameter {param_name} should affect cache version hash"


def test_comprehensive_no_hash_collisions(create_dummy_kitti_structure):
    """Ensure no hash collisions across many different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kitti_structure(temp_dir)
        
        datasets = []
        
        # Generate different dataset configurations
        for split in ['train', 'val', 'test']:
            for base_seed_val in [None, 42, 123]:
                datasets.append(KITTIDataset(
                    data_root=temp_dir,
                    split=split,
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


