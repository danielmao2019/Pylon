"""Tests for MultiTaskFacialLandmarkDataset cache version discrimination."""

import pytest
import tempfile
import os
import numpy as np
from PIL import Image
from data.datasets.multi_task_datasets.multi_task_facial_landmark_dataset import MultiTaskFacialLandmarkDataset


def create_dummy_facial_landmark_structure(data_root: str) -> None:
    """Create a dummy MultiTaskFacialLandmark directory structure for testing."""
    # Define sample data
    train_data = [
        ('train_images/img001.jpg', [1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], 1, 0, 1, 2),
        ('train_images/img002.jpg', [2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0], 0, 1, 0, 1),
        ('train_images/img003.jpg', [3.0, 4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0, 12.0], 1, 1, 1, 0),
    ]
    
    test_data = [
        ('test_images/img004.jpg', [4.0, 5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0, 13.0], 0, 0, 0, 2),
        ('test_images/img005.jpg', [5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0], 1, 0, 1, 1),
    ]
    
    # Create directory structure
    os.makedirs(os.path.join(data_root, 'train_images'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'test_images'), exist_ok=True)
    
    # Create training.txt file
    with open(os.path.join(data_root, 'training.txt'), 'w') as f:
        for img_path, landmarks_x, landmarks_y, gender, smile, glasses, pose in train_data:
            # Create dummy image
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(data_root, img_path))
            
            # Write annotation line
            line_parts = [img_path] + [str(x) for x in landmarks_x] + [str(y) for y in landmarks_y] + [str(gender), str(smile), str(glasses), str(pose)]
            f.write(' '.join(line_parts) + '\n')
    
    # Create testing.txt file
    with open(os.path.join(data_root, 'testing.txt'), 'w') as f:
        for img_path, landmarks_x, landmarks_y, gender, smile, glasses, pose in test_data:
            # Create dummy image
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(data_root, img_path))
            
            # Write annotation line
            line_parts = [img_path] + [str(x) for x in landmarks_x] + [str(y) for y in landmarks_y] + [str(gender), str(smile), str(glasses), str(pose)]
            f.write(' '.join(line_parts) + '\n')


def test_multi_task_facial_landmark_dataset_version_discrimination():
    """Test that MultiTaskFacialLandmarkDataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_facial_landmark_structure(temp_dir)
        
        # Same parameters should have same hash
        dataset1a = MultiTaskFacialLandmarkDataset(
            data_root=temp_dir,
            split='train'
        )
        dataset1b = MultiTaskFacialLandmarkDataset(
            data_root=temp_dir,
            split='train'
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different split should have different hash
        dataset2 = MultiTaskFacialLandmarkDataset(
            data_root=temp_dir,
            split='test'  # Different
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different data_root should have SAME hash (data_root excluded from versioning)
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_facial_landmark_structure(temp_dir2)
            dataset3 = MultiTaskFacialLandmarkDataset(
                data_root=temp_dir2,  # Different path, same content structure
                split='train'
            )
            assert dataset1a.get_cache_version_hash() == dataset3.get_cache_version_hash()


def test_split_variants():
    """Test that different splits produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_facial_landmark_structure(temp_dir)
        
        split_variants = ['train', 'test']
        
        datasets = []
        for split in split_variants:
            dataset = MultiTaskFacialLandmarkDataset(
                data_root=temp_dir,
                split=split
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All split variants should produce different hashes, got: {hashes}"


def test_different_selective_loading_configurations():
    """Test that datasets with different selective loading configurations have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_facial_landmark_structure(temp_dir)
        
        # Create datasets with different selective loading configurations
        dataset1 = MultiTaskFacialLandmarkDataset(
            data_root=temp_dir,
            split='train',
            labels=['landmarks', 'gender']
        )
        
        dataset2 = MultiTaskFacialLandmarkDataset(
            data_root=temp_dir,
            split='train',
            labels=['smile', 'glasses', 'pose']
        )
        
        dataset3 = MultiTaskFacialLandmarkDataset(
            data_root=temp_dir,
            split='train'  # No labels parameter (loads all)
        )
        
        # Should have different hashes due to different selective loading configurations
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()
        assert dataset1.get_cache_version_hash() != dataset3.get_cache_version_hash()
        assert dataset2.get_cache_version_hash() != dataset3.get_cache_version_hash()


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across different dataset configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_facial_landmark_structure(temp_dir)
        
        datasets = []
        
        # Generate different dataset configurations
        # Note: data_root is intentionally ignored in cache versioning to allow cache sharing
        for split in ['train', 'test']:
            # Test different splits
            datasets.append(MultiTaskFacialLandmarkDataset(
                data_root=temp_dir,
                split=split
            ))
            
            # Test different selective loading configurations
            datasets.append(MultiTaskFacialLandmarkDataset(
                data_root=temp_dir,
                split=split,
                labels=['landmarks']
            ))
            
            datasets.append(MultiTaskFacialLandmarkDataset(
                data_root=temp_dir,
                split=split,
                labels=['gender', 'smile']
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
