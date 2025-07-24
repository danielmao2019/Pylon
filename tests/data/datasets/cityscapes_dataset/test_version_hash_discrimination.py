"""Tests for CityScapesDataset cache version discrimination."""

import pytest
import tempfile
import os
import numpy as np
from contextlib import contextmanager
from PIL import Image
from data.datasets.multi_task_datasets.city_scapes_dataset import CityScapesDataset


@contextmanager
def patched_dataset_size():
    """Context manager to temporarily patch DATASET_SIZE for testing."""
    original_dataset_size = CityScapesDataset.DATASET_SIZE
    CityScapesDataset.DATASET_SIZE = {
        'train': 6,  # 2 cities * 3 images each
        'val': 6,    # 2 cities * 3 images each
    }
    try:
        yield
    finally:
        CityScapesDataset.DATASET_SIZE = original_dataset_size


def create_dummy_cityscapes_structure(data_root: str) -> None:
    """Create a dummy CityScapes directory structure for testing."""
    # Define cities for train and val splits
    train_cities = ['aachen', 'bochum']
    val_cities = ['frankfurt', 'lindau']
    
    # Create directory structure for each split
    for split, cities in [('train', train_cities), ('val', val_cities)]:
        # Create directories for each data type
        leftimg_dir = os.path.join(data_root, 'leftImg8bit', split)
        disparity_dir = os.path.join(data_root, 'disparity', split)
        gtfine_dir = os.path.join(data_root, 'gtFine', split)
        
        for city in cities:
            os.makedirs(os.path.join(leftimg_dir, city), exist_ok=True)
            os.makedirs(os.path.join(disparity_dir, city), exist_ok=True)
            os.makedirs(os.path.join(gtfine_dir, city), exist_ok=True)
            
            # Create 3 dummy images per city
            for i in range(3):
                img_name = f"{city}_000000_00000{i}"
                
                # Create RGB image
                img = np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
                Image.fromarray(img).save(
                    os.path.join(leftimg_dir, city, f"{img_name}_leftImg8bit.png")
                )
                
                # Create disparity map (16-bit depth values)
                disparity = np.random.randint(0, 5000, (1024, 2048), dtype=np.uint16)
                Image.fromarray(disparity).save(
                    os.path.join(disparity_dir, city, f"{img_name}_disparity.png")
                )
                
                # Create semantic segmentation label
                # Use valid class IDs from the dataset
                valid_ids = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
                semantic = np.random.choice(valid_ids, size=(1024, 2048), replace=True).astype(np.uint8)
                Image.fromarray(semantic).save(
                    os.path.join(gtfine_dir, city, f"{img_name}_gtFine_labelIds.png")
                )
                
                # Create instance segmentation label
                instance = np.random.randint(0, 1000, (1024, 2048), dtype=np.uint16)
                Image.fromarray(instance).save(
                    os.path.join(gtfine_dir, city, f"{img_name}_gtFine_instanceIds.png")
                )


def test_cityscapes_dataset_version_discrimination():
    """Test that CityScapesDataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_cityscapes_structure(temp_dir)
        
        with patched_dataset_size():
            # Same parameters should have same hash
            dataset1a = CityScapesDataset(
                data_root=temp_dir,
                split='train',
                semantic_granularity='coarse'
            )
        dataset1b = CityScapesDataset(
            data_root=temp_dir,
            split='train',
            semantic_granularity='coarse'
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different split should have different hash
        dataset2 = CityScapesDataset(
            data_root=temp_dir,
            split='val',  # Different
            semantic_granularity='coarse'
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different semantic_granularity should have different hash
        dataset3 = CityScapesDataset(
            data_root=temp_dir,
            split='train',
            semantic_granularity='fine'  # Different
        )
        assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()
        
        # Different data_root should have different hash
        with tempfile.TemporaryDirectory() as temp_dir2:
            create_dummy_cityscapes_structure(temp_dir2)
            dataset4 = CityScapesDataset(
                data_root=temp_dir2,  # Different
                split='train',
                semantic_granularity='coarse'
            )
            assert dataset1a.get_cache_version_hash() != dataset4.get_cache_version_hash()


def test_split_variants():
    """Test that different splits produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_cityscapes_structure(temp_dir)
        
        split_variants = ['train', 'val']
        
        datasets = []
        for split in split_variants:
            dataset = CityScapesDataset(
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
        create_dummy_cityscapes_structure(temp_dir)
        
        granularity_variants = ['fine', 'coarse']
        
        datasets = []
        for granularity in granularity_variants:
            dataset = CityScapesDataset(
                data_root=temp_dir,
                split='train',
                semantic_granularity=granularity
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All granularity variants should produce different hashes, got: {hashes}"


def test_different_file_structure():
    """Test that different file structures produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_cityscapes_structure(temp_dir)
        
        dataset1 = CityScapesDataset(
            data_root=temp_dir,
            split='train',
            semantic_granularity='coarse'
        )
        
        # Add another city to the train split
        new_city = 'cologne'
        leftimg_dir = os.path.join(temp_dir, 'leftImg8bit', 'train', new_city)
        disparity_dir = os.path.join(temp_dir, 'disparity', 'train', new_city)
        gtfine_dir = os.path.join(temp_dir, 'gtFine', 'train', new_city)
        
        os.makedirs(leftimg_dir, exist_ok=True)
        os.makedirs(disparity_dir, exist_ok=True)
        os.makedirs(gtfine_dir, exist_ok=True)
        
        # Add one image to the new city
        img_name = f"{new_city}_000000_000000"
        img = np.zeros((1024, 2048, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(leftimg_dir, f"{img_name}_leftImg8bit.png"))
        
        dataset2 = CityScapesDataset(
            data_root=temp_dir,
            split='train',
            semantic_granularity='coarse'
        )
        
        # Should have different hashes due to different file structure
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_inherited_parameters_affect_version_hash():
    """Test that parameters inherited from BaseDataset affect version hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_cityscapes_structure(temp_dir)
        
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
        
        dataset1 = CityScapesDataset(**base_args)
        
        for param_name, new_value in parameter_variants:
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = CityScapesDataset(**modified_args)
            
            assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
                f"Inherited parameter {param_name} should affect cache version hash"


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_cityscapes_structure(temp_dir)
        
        datasets = []
        
        # Generate different dataset configurations
        for split in ['train', 'val']:
            for granularity in ['fine', 'coarse']:
                for initial_seed in [None, 42, 123]:
                    datasets.append(CityScapesDataset(
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


