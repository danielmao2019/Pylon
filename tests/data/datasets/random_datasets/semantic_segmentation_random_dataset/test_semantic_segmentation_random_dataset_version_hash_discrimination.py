"""Tests for SemanticSegmentationRandomDataset cache version discrimination."""

import pytest
from data.datasets.random_datasets.semantic_segmentation_random_dataset import SemanticSegmentationRandomDataset


def test_semantic_segmentation_random_dataset_version_discrimination():
    """Test that SemanticSegmentationRandomDataset instances with different parameters have different hashes."""
    
    # Same parameters should have same hash
    dataset1a = SemanticSegmentationRandomDataset(
        num_classes=10,
        num_examples=100,
        initial_seed=42
    )
    dataset1b = SemanticSegmentationRandomDataset(
        num_classes=10,
        num_examples=100,
        initial_seed=42
    )
    assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
    
    # Different num_classes should have different hash
    dataset2 = SemanticSegmentationRandomDataset(
        num_classes=20,  # Different
        num_examples=100,
        initial_seed=42
    )
    assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
    
    # Different num_examples should have different hash
    dataset3 = SemanticSegmentationRandomDataset(
        num_classes=10,
        num_examples=200,  # Different
        initial_seed=42
    )
    assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()
    
    # Different initial_seed should have different hash
    dataset4 = SemanticSegmentationRandomDataset(
        num_classes=10,
        num_examples=100,
        initial_seed=123  # Different
    )
    assert dataset1a.get_cache_version_hash() != dataset4.get_cache_version_hash()


def test_all_parameters_affect_version_hash():
    """Test that all content-affecting parameters impact the version hash."""
    
    base_args = {
        'num_classes': 10,
        'num_examples': 100,
        'initial_seed': 42,
    }
    
    # Test each parameter individually
    parameter_variants = [
        ('num_classes', 5),
        ('num_classes', 20),
        ('num_classes', 100),  # Large number affects mask size
        ('num_examples', 50),
        ('num_examples', 200),
        ('initial_seed', 123),
        ('initial_seed', None),  # None vs specified
    ]
    
    dataset1 = SemanticSegmentationRandomDataset(**base_args)
    
    for param_name, new_value in parameter_variants:
        modified_args = base_args.copy()
        modified_args[param_name] = new_value
        dataset2 = SemanticSegmentationRandomDataset(**modified_args)
        
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
            f"Parameter {param_name} should affect cache version hash"


def test_none_vs_specified_initial_seed():
    """Test that None vs specified initial_seed produces different hashes."""
    
    # None vs specified should have different hashes
    dataset1 = SemanticSegmentationRandomDataset(
        num_classes=10,
        num_examples=100,
        initial_seed=None
    )
    dataset2 = SemanticSegmentationRandomDataset(
        num_classes=10,
        num_examples=100,
        initial_seed=42
    )
    assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_num_classes_affects_mask_size():
    """Test that different numbers of classes (which affect mask size) are properly discriminated."""
    
    # num_classes affects the mask size through math.ceil(math.sqrt(num_classes))
    # Test cases that produce different mask sizes
    num_classes_variants = [
        4,   # mask size: (2, 2)
        9,   # mask size: (3, 3)
        16,  # mask size: (4, 4)
        25,  # mask size: (5, 5)
        30,  # mask size: (6, 6) - ceil(sqrt(30)) = ceil(5.477) = 6
        36,  # mask size: (6, 6) - ceil(sqrt(36)) = 6
        49,  # mask size: (7, 7)
    ]
    
    datasets = []
    for num_classes in num_classes_variants:
        dataset = SemanticSegmentationRandomDataset(
            num_classes=num_classes,
            num_examples=100,
            initial_seed=42
        )
        datasets.append(dataset)
    
    # All should have different hashes
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    assert len(hashes) == len(set(hashes)), \
        f"All num_classes variants should produce different hashes, got: {hashes}"


def test_edge_case_num_classes():
    """Test edge cases for num_classes values."""
    
    edge_case_variants = [
        1,    # Minimum valid value, mask size: (1, 1)
        2,    # mask size: (2, 2)
        3,    # mask size: (2, 2) - same as 2, but different high value
        100,  # mask size: (10, 10)
        101,  # mask size: (11, 11)
    ]
    
    datasets = []
    for num_classes in edge_case_variants:
        dataset = SemanticSegmentationRandomDataset(
            num_classes=num_classes,
            num_examples=100,
            initial_seed=42
        )
        datasets.append(dataset)
    
    # All should have different hashes (even cases with same mask size should differ due to 'high' parameter)
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    assert len(hashes) == len(set(hashes)), \
        f"All edge case num_classes variants should produce different hashes, got: {hashes}"


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    
    datasets = []
    
    # Generate many different dataset configurations
    for num_classes in [4, 9, 16, 25, 30]:  # Different mask sizes
        for num_examples in [50, 100, 200]:
            for initial_seed in [None, 42, 123]:
                datasets.append(SemanticSegmentationRandomDataset(
                    num_classes=num_classes,
                    num_examples=num_examples,
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


