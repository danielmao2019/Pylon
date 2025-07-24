"""Tests for ClassificationRandomDataset cache version discrimination."""

import pytest
import tempfile
from data.datasets.random_datasets.classification_random_dataset import ClassificationRandomDataset


def test_classification_random_dataset_version_discrimination():
    """Test that ClassificationRandomDataset instances with different parameters have different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Same parameters should have same hash
        dataset1a = ClassificationRandomDataset(
            num_classes=10,
            num_examples=100,
            image_res=(224, 224),
            initial_seed=42,
            data_root=temp_dir,
            split='all'
        )
        dataset1b = ClassificationRandomDataset(
            num_classes=10,
            num_examples=100,
            image_res=(224, 224),
            initial_seed=42,
            data_root=temp_dir,
            split='all'
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
        
        # Different num_classes should have different hash
        dataset2 = ClassificationRandomDataset(
            num_classes=20,  # Different
            num_examples=100,
            image_res=(224, 224),
            initial_seed=42,
            data_root=temp_dir,
            split='all'
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different num_examples should have different hash
        dataset3 = ClassificationRandomDataset(
            num_classes=10,
            num_examples=200,  # Different
            image_res=(224, 224),
            initial_seed=42,
            data_root=temp_dir,
            split='all'
        )
        assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()
        
        # Different image_res should have different hash
        dataset4 = ClassificationRandomDataset(
            num_classes=10,
            num_examples=100,
            image_res=(256, 256),  # Different
            initial_seed=42,
            data_root=temp_dir,
            split='all'
        )
        assert dataset1a.get_cache_version_hash() != dataset4.get_cache_version_hash()
        
        # Different initial_seed should have different hash
        dataset5 = ClassificationRandomDataset(
            num_classes=10,
            num_examples=100,
            image_res=(224, 224),
            initial_seed=123,  # Different
            data_root=temp_dir,
            split='all'
        )
        assert dataset1a.get_cache_version_hash() != dataset5.get_cache_version_hash()


def test_all_parameters_affect_version_hash():
    """Test that all content-affecting parameters impact the version hash."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_args = {
            'num_classes': 10,
            'num_examples': 100,
            'image_res': (224, 224),
            'initial_seed': 42,
            'data_root': temp_dir,
            'split': 'all',
        }
        
        # Test each parameter individually
        parameter_variants = [
            ('num_classes', 5),
            ('num_classes', 20),
            ('num_examples', 50),
            ('num_examples', 200),
            ('image_res', (128, 128)),
            ('image_res', (256, 256)),
            ('image_res', (224, 512)),  # Different aspect ratio
            ('initial_seed', 123),
            ('initial_seed', None),  # None vs specified
        ]
        
        dataset1 = ClassificationRandomDataset(**base_args)
        
        for param_name, new_value in parameter_variants:
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = ClassificationRandomDataset(**modified_args)
            
            assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
                f"Parameter {param_name} should affect cache version hash"


def test_none_vs_specified_initial_seed():
    """Test that None vs specified initial_seed produces different hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # None vs specified should have different hashes
        dataset1 = ClassificationRandomDataset(
            num_classes=10,
            num_examples=100,
            image_res=(224, 224),
            initial_seed=None,
            data_root=temp_dir,
            split='all'
        )
        dataset2 = ClassificationRandomDataset(
            num_classes=10,
            num_examples=100,
            image_res=(224, 224),
            initial_seed=42,
            data_root=temp_dir,
            split='all'
        )
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_image_resolution_variants():
    """Test that different image resolutions are properly discriminated."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        resolution_variants = [
            (224, 224),
            (256, 256),
            (128, 128),
            (224, 512),  # Different aspect ratio
            (512, 224),  # Different aspect ratio
            (299, 299),  # Inception size
        ]
        
        datasets = []
        for image_res in resolution_variants:
            dataset = ClassificationRandomDataset(
                num_classes=10,
                num_examples=100,
                image_res=image_res,
                initial_seed=42,
                data_root=temp_dir,
                split='all'
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All image resolution variants should produce different hashes, got: {hashes}"


def test_num_classes_variants():
    """Test that different numbers of classes are properly discriminated."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        num_classes_variants = [2, 5, 10, 20, 100, 1000]
        
        datasets = []
        for num_classes in num_classes_variants:
            dataset = ClassificationRandomDataset(
                num_classes=num_classes,
                num_examples=100,
                image_res=(224, 224),
                initial_seed=42,
                data_root=temp_dir,
                split='all'
            )
            datasets.append(dataset)
        
        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All num_classes variants should produce different hashes, got: {hashes}"


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        datasets = []
        
        # Generate many different dataset configurations
        for num_classes in [5, 10, 20]:
            for num_examples in [50, 100, 200]:
                for image_res in [(224, 224), (256, 256), (128, 128)]:
                    for initial_seed in [None, 42, 123]:
                        datasets.append(ClassificationRandomDataset(
                            num_classes=num_classes,
                            num_examples=num_examples,
                            image_res=image_res,
                            initial_seed=initial_seed,
                            data_root=temp_dir,
                            split='all'
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


