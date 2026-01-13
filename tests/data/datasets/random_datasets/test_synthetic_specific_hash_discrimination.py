"""Synthetic/Random dataset-specific hash discrimination tests.

This module contains tests specific to synthetic/random dataset parameters that don't
apply to other datasets, particularly num_examples, num_classes, and other
synthetic-specific parameters.
"""

import pytest
from utils.builders.builder import build_from_config


@pytest.mark.parametrize('random_dataset_class,base_args', [
    ('ClassificationRandomDataset', {'num_classes': 10, 'image_res': (32, 32)}),
    ('SemanticSegmentationRandomDataset', {'num_classes': 21}),
])
def test_synthetic_num_examples_discrimination(random_dataset_class, base_args):
    """Test synthetic dataset num_examples parameter discrimination."""
    # Import the appropriate class
    if random_dataset_class == 'ClassificationRandomDataset':
        from data.datasets.random_datasets.classification_random_dataset import ClassificationRandomDataset
        dataset_class = ClassificationRandomDataset
    else:
        from data.datasets.random_datasets.semantic_segmentation_random_dataset import SemanticSegmentationRandomDataset
        dataset_class = SemanticSegmentationRandomDataset

    # Test num_examples variation
    config1 = {
        'class': dataset_class,
        'args': {**base_args, 'num_examples': 50}
    }
    config2 = {
        'class': dataset_class,
        'args': {**base_args, 'num_examples': 100}
    }

    dataset1 = build_from_config(config1)
    dataset2 = build_from_config(config2)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 != hash2, f"{random_dataset_class}: Different num_examples should produce different hashes: {hash1} == {hash2}"


@pytest.mark.parametrize('random_dataset_class,base_args', [
    ('ClassificationRandomDataset', {'num_examples': 50, 'image_res': (32, 32)}),
    ('SemanticSegmentationRandomDataset', {'num_examples': 50}),
])
def test_synthetic_num_classes_discrimination(random_dataset_class, base_args):
    """Test synthetic dataset num_classes parameter discrimination."""
    # Import the appropriate class
    if random_dataset_class == 'ClassificationRandomDataset':
        from data.datasets.random_datasets.classification_random_dataset import ClassificationRandomDataset
        dataset_class = ClassificationRandomDataset
        class_variations = [10, 20]
    else:
        from data.datasets.random_datasets.semantic_segmentation_random_dataset import SemanticSegmentationRandomDataset
        dataset_class = SemanticSegmentationRandomDataset
        class_variations = [21, 151]

    # Test num_classes variation
    config1 = {
        'class': dataset_class,
        'args': {**base_args, 'num_classes': class_variations[0]}
    }
    config2 = {
        'class': dataset_class,
        'args': {**base_args, 'num_classes': class_variations[1]}
    }

    dataset1 = build_from_config(config1)
    dataset2 = build_from_config(config2)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 != hash2, f"{random_dataset_class}: Different num_classes should produce different hashes: {hash1} == {hash2}"


def test_classification_random_image_res_discrimination():
    """Test ClassificationRandomDataset image_res parameter discrimination."""
    from data.datasets.random_datasets.classification_random_dataset import ClassificationRandomDataset

    base_config = {
        'num_examples': 50,
        'num_classes': 10,
    }

    # Test image_res variation
    config1 = {
        'class': ClassificationRandomDataset,
        'args': {**base_config, 'image_res': (32, 32)}
    }
    config2 = {
        'class': ClassificationRandomDataset,
        'args': {**base_config, 'image_res': (64, 64)}
    }

    dataset1 = build_from_config(config1)
    dataset2 = build_from_config(config2)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 != hash2, f"ClassificationRandomDataset: Different image_res should produce different hashes: {hash1} == {hash2}"


def test_semantic_segmentation_random_initial_seed_discrimination():
    """Test SemanticSegmentationRandomDataset initial_seed parameter discrimination."""
    from data.datasets.random_datasets.semantic_segmentation_random_dataset import SemanticSegmentationRandomDataset

    base_config = {
        'num_examples': 50,
        'num_classes': 21,
    }

    # Test initial_seed variation (None vs specific value)
    config1 = {
        'class': SemanticSegmentationRandomDataset,
        'args': {**base_config, 'base_seed': None}
    }
    config2 = {
        'class': SemanticSegmentationRandomDataset,
        'args': {**base_config, 'base_seed': 42}
    }

    dataset1 = build_from_config(config1)
    dataset2 = build_from_config(config2)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 != hash2, f"SemanticSegmentationRandomDataset: Different initial_seed should produce different hashes: {hash1} == {hash2}"


def test_synthetic_comprehensive_parameter_combinations():
    """Test comprehensive synthetic parameter combinations produce unique hashes."""
    from data.datasets.random_datasets.classification_random_dataset import ClassificationRandomDataset

    test_combinations = [
        {'num_examples': 50, 'num_classes': 10, 'image_res': (32, 32)},
        {'num_examples': 100, 'num_classes': 10, 'image_res': (32, 32)},
        {'num_examples': 50, 'num_classes': 20, 'image_res': (32, 32)},
        {'num_examples': 50, 'num_classes': 10, 'image_res': (64, 64)},
    ]

    hashes = []
    for combination in test_combinations:
        config = {
            'class': ClassificationRandomDataset,
            'args': combination
        }
        dataset = build_from_config(config)
        hash_val = dataset.get_cache_version_hash()

        # Check for collision
        assert hash_val not in hashes, f"Hash collision detected for {combination}: hash {hash_val} already exists"
        hashes.append(hash_val)

    # Verify we have unique hashes
    assert len(hashes) == len(test_combinations), f"Expected {len(test_combinations)} unique hashes, got {len(hashes)}"