"""Tests for NYUv2Dataset cache version discrimination."""

import pytest
import tempfile
import os
import numpy as np
import scipy.io
from PIL import Image
from data.datasets.multi_task_datasets.nyu_v2_dataset import NYUv2Dataset


def create_dummy_nyuv2_structure(data_root: str) -> None:
    """Create a dummy NYU-v2 directory structure for testing."""
    # Create directory structure
    os.makedirs(os.path.join(data_root, 'gt_sets'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'normals'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'segmentation'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'edge'), exist_ok=True)
    
    # Define sample IDs for train and val
    train_ids = ['img_0001', 'img_0002', 'img_0003']
    val_ids = ['img_0004', 'img_0005']
    
    # Create gt_sets files
    with open(os.path.join(data_root, 'gt_sets', 'train.txt'), 'w') as f:
        for img_id in train_ids:
            f.write(f"{img_id}\n")
    
    with open(os.path.join(data_root, 'gt_sets', 'val.txt'), 'w') as f:
        for img_id in val_ids:
            f.write(f"{img_id}\n")
    
    # Create dummy data for all IDs
    all_ids = train_ids + val_ids
    for img_id in all_ids:
        # Create RGB image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(data_root, 'images', f"{img_id}.jpg"))
        
        # Create depth map (.mat file)
        depth_data = np.random.rand(480, 640).astype(np.float32) * 10.0  # Depth values 0-10m
        scipy.io.savemat(
            os.path.join(data_root, 'depth', f"{img_id}.mat"),
            {'depth': depth_data}
        )
        
        # Create normal map (as RGB image)
        normal = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        Image.fromarray(normal).save(os.path.join(data_root, 'normals', f"{img_id}.jpg"))
        
        # Create semantic segmentation (.mat file)
        segmentation = np.random.randint(0, 41, (480, 640), dtype=np.uint8)
        scipy.io.savemat(
            os.path.join(data_root, 'segmentation', f"{img_id}.mat"),
            {'segmentation': segmentation}
        )
        
        # Create edge map
        edge = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        Image.fromarray(edge).save(os.path.join(data_root, 'edge', f"{img_id}.png"))


def test_nyuv2_dataset_version_discrimination():
    """Test that NYUv2Dataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_nyuv2_structure(temp_dir)
        
        # Same parameters should have same hash
        dataset1a = NYUv2Dataset(
            data_root=temp_dir,
            split='train',
            semantic_granularity='coarse'
        )
        dataset1b = NYUv2Dataset(
            data_root=temp_dir,
            split='train',
            semantic_granularity='coarse'
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different split should have different hash
        dataset2 = NYUv2Dataset(
            data_root=temp_dir,
            split='val',  # Different
            semantic_granularity='coarse'
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different semantic_granularity should have different hash
        dataset3 = NYUv2Dataset(
            data_root=temp_dir,
            split='train',
            semantic_granularity='fine'  # Different
        )
        assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()
        
        # Different data_root should have different hash
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_nyuv2_structure(temp_dir2)
            dataset4 = NYUv2Dataset(
                data_root=temp_dir2,  # Different
                split='train',
                semantic_granularity='coarse'
            )
            assert dataset1a.get_cache_version_hash() != dataset4.get_cache_version_hash()


def test_split_variants():
    """Test that different splits produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_nyuv2_structure(temp_dir)
        
        split_variants = ['train', 'val']
        
        datasets = []
        for split in split_variants:
            dataset = NYUv2Dataset(
                data_root=temp_dir,
                split=split,
                semantic_granularity='coarse'
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All split variants should produce different hashes, got: {hashes}"


def test_semantic_granularity_variants():
    """Test that different semantic granularities produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_nyuv2_structure(temp_dir)
        
        granularity_variants = ['fine', 'coarse']
        
        datasets = []
        for granularity in granularity_variants:
            dataset = NYUv2Dataset(
                data_root=temp_dir,
                split='train',
                semantic_granularity=granularity
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All granularity variants should produce different hashes, got: {hashes}"


def test_different_annotation_data():
    """Test that different annotation data produces different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_nyuv2_structure(temp_dir)
        
        dataset1 = NYUv2Dataset(
            data_root=temp_dir,
            split='train',
            semantic_granularity='coarse'
        )
        
        # Modify the gt_sets file
        gt_sets_file = os.path.join(temp_dir, 'gt_sets', 'train.txt')
        with open(gt_sets_file, 'a') as f:
            f.write("img_0099\n")  # Add new ID
            
        # Create corresponding dummy files for the new ID
        img_id = 'img_0099'
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(temp_dir, 'images', f"{img_id}.jpg"))
        
        depth_data = np.zeros((480, 640), dtype=np.float32)
        scipy.io.savemat(os.path.join(temp_dir, 'depth', f"{img_id}.mat"), {'depth': depth_data})
        
        Image.fromarray(img).save(os.path.join(temp_dir, 'normals', f"{img_id}.jpg"))
        
        segmentation = np.zeros((480, 640), dtype=np.uint8)
        scipy.io.savemat(os.path.join(temp_dir, 'segmentation', f"{img_id}.mat"), {'segmentation': segmentation})
        
        edge = np.zeros((480, 640), dtype=np.uint8)
        Image.fromarray(edge).save(os.path.join(temp_dir, 'edge', f"{img_id}.png"))
        
        dataset2 = NYUv2Dataset(
            data_root=temp_dir,
            split='train',
            semantic_granularity='coarse'
        )
        
        # Should have different hashes due to different annotation list
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_inherited_parameters_affect_version_hash():
    """Test that parameters inherited from BaseDataset affect version hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_nyuv2_structure(temp_dir)
        
        base_args = {
            'data_root': temp_dir,
            'split': 'train',
            'semantic_granularity': 'coarse',
        }
        
        # Test inherited parameters from BaseDataset
        parameter_variants = [
            ('initial_seed', 42),  # Different from default None
            ('cache_size', 1000),  # Different from default
        ]
        
        dataset1 = NYUv2Dataset(**base_args)
        
        for param_name, new_value in parameter_variants:
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = NYUv2Dataset(**modified_args)
            
            assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
                f"Inherited parameter {param_name} should affect cache version hash"


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_nyuv2_structure(temp_dir)
        
        datasets = []
        
        # Generate different dataset configurations
        for split in ['train', 'val']:
            for granularity in ['fine', 'coarse']:
                for initial_seed in [None, 42, 123]:
                    datasets.append(NYUv2Dataset(
                        data_root=temp_dir,
                        split=split,
                        semantic_granularity=granularity,
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


