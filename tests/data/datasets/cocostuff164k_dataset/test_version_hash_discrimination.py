"""Tests for COCOStuff164KDataset cache version discrimination."""

import pytest
import tempfile
import os
import numpy as np
from contextlib import contextmanager
from PIL import Image
from data.datasets.semantic_segmentation_datasets.coco_stuff_164k_dataset import COCOStuff164KDataset


@contextmanager
def patched_dataset_size():
    """Context manager to temporarily patch DATASET_SIZE for testing."""
    original_dataset_size = COCOStuff164KDataset.DATASET_SIZE
    COCOStuff164KDataset.DATASET_SIZE = {
        'train2017': 3,
        'val2017': 2,
    }
    try:
        yield
    finally:
        COCOStuff164KDataset.DATASET_SIZE = original_dataset_size


def create_dummy_cocostuff_structure(data_root: str) -> None:
    """Create a dummy COCO-Stuff164K directory structure for testing."""
    # Create directory structure
    images_train_dir = os.path.join(data_root, 'images', 'train2017')
    images_val_dir = os.path.join(data_root, 'images', 'val2017')
    annotations_train_dir = os.path.join(data_root, 'annotations', 'train2017')
    annotations_val_dir = os.path.join(data_root, 'annotations', 'val2017')
    curated_train_dir = os.path.join(data_root, 'curated', 'train2017')
    curated_val_dir = os.path.join(data_root, 'curated', 'val2017')
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(annotations_train_dir, exist_ok=True)
    os.makedirs(annotations_val_dir, exist_ok=True)
    os.makedirs(curated_train_dir, exist_ok=True)
    os.makedirs(curated_val_dir, exist_ok=True)
    
    # Create dummy image IDs
    train_ids = ['000000000001', '000000000002', '000000000003']
    val_ids = ['000000000004', '000000000005']
    
    # Create curated file lists
    with open(os.path.join(curated_train_dir, 'Coco164kFull_Stuff_Coarse.txt'), 'w') as f:
        for img_id in train_ids:
            f.write(f"{img_id}\n")
    
    with open(os.path.join(curated_val_dir, 'Coco164kFull_Stuff_Coarse.txt'), 'w') as f:
        for img_id in val_ids:
            f.write(f"{img_id}\n")
    
    # Create dummy images and annotations
    for img_id in train_ids:
        # Create RGB image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(images_train_dir, f"{img_id}.jpg"))
        
        # Create segmentation label (values 0-181 for fine labels)
        label = np.random.randint(0, 182, (256, 256), dtype=np.uint8)
        Image.fromarray(label).save(os.path.join(annotations_train_dir, f"{img_id}.png"))
    
    for img_id in val_ids:
        # Create RGB image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(images_val_dir, f"{img_id}.jpg"))
        
        # Create segmentation label
        label = np.random.randint(0, 182, (256, 256), dtype=np.uint8)
        Image.fromarray(label).save(os.path.join(annotations_val_dir, f"{img_id}.png"))


def test_cocostuff164k_dataset_version_discrimination():
    """Test that COCOStuff164KDataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_cocostuff_structure(temp_dir)
        
        with patched_dataset_size():
            # Same parameters should have same hash
            dataset1a = COCOStuff164KDataset(
                data_root=temp_dir,
                split='train2017',
                semantic_granularity='coarse'
            )
            dataset1b = COCOStuff164KDataset(
                data_root=temp_dir,
                split='train2017',
                semantic_granularity='coarse'
            )
            assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
            
            # Different split should have different hash
            dataset2 = COCOStuff164KDataset(
                data_root=temp_dir,
                split='val2017',  # Different
                semantic_granularity='coarse'
            )
            assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
            
            # Note: semantic_granularity doesn't affect cache version hash as it's
            # considered a processing parameter, not a content parameter
            
            # Different data_root should have different hash
            with tempfile.TemporaryDirectory() as temp_dir2:
                create_dummy_cocostuff_structure(temp_dir2)
                dataset4 = COCOStuff164KDataset(
                    data_root=temp_dir2,  # Different
                    split='train2017',
                    semantic_granularity='coarse'
                )
                assert dataset1a.get_cache_version_hash() != dataset4.get_cache_version_hash()


def test_split_variants():
    """Test that different splits produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_cocostuff_structure(temp_dir)
        
        with patched_dataset_size():
            split_variants = ['train2017', 'val2017']
            
            datasets = []
            for split in split_variants:
                dataset = COCOStuff164KDataset(
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
    """Test that semantic_granularity parameter works but doesn't affect cache version."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_cocostuff_structure(temp_dir)
        
        with patched_dataset_size():
            # Both granularities should have same cache version hash since it's a processing parameter
            dataset_fine = COCOStuff164KDataset(
                data_root=temp_dir,
                split='train2017',
                semantic_granularity='fine'
            )
            dataset_coarse = COCOStuff164KDataset(
                data_root=temp_dir,
                split='train2017',
                semantic_granularity='coarse'
            )
            
            # Should have same hash since semantic_granularity is processing-only
            assert dataset_fine.get_cache_version_hash() == dataset_coarse.get_cache_version_hash()
            
            # But they should have different NUM_CLASSES
            assert dataset_fine.NUM_CLASSES != dataset_coarse.NUM_CLASSES


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across different split configurations."""
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_cocostuff_structure(temp_dir1)
            create_dummy_cocostuff_structure(temp_dir2)
            
            with patched_dataset_size():
                datasets = []
                
                # Generate different dataset configurations
                for data_root in [temp_dir1, temp_dir2]:
                    for split in ['train2017', 'val2017']:
                        datasets.append(COCOStuff164KDataset(
                            data_root=data_root,
                            split=split,
                            semantic_granularity='coarse'  # Keep this constant
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


