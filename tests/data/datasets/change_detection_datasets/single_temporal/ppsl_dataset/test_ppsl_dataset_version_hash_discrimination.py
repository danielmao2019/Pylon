"""Test version hash discrimination for PPSLDataset.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
from data.datasets.change_detection_datasets.single_temporal.ppsl_dataset import PPSLDataset
from data.datasets.random_datasets.semantic_segmentation_random_dataset import SemanticSegmentationRandomDataset


def test_ppsl_same_parameters_same_hash():
    """Test that identical parameters produce identical hashes."""

    # Create identical source datasets
    source1a = SemanticSegmentationRandomDataset(
        num_examples=10,
        num_classes=5,
        base_seed=42
    )
    source1b = SemanticSegmentationRandomDataset(
        num_examples=10,
        num_classes=5,
        base_seed=42
    )

    # Same parameters should produce same hash
    dataset1a = PPSLDataset(source=source1a, dataset_size=10)
    dataset1b = PPSLDataset(source=source1b, dataset_size=10)

    hash1a = dataset1a.get_cache_version_hash()
    hash1b = dataset1b.get_cache_version_hash()

    assert hash1a == hash1b, f"Same parameters should produce same hash: {hash1a} != {hash1b}"


def test_ppsl_different_source_different_hash():
    """Test that different source datasets produce different hashes."""

    # Create different source datasets
    source1 = SemanticSegmentationRandomDataset(
        num_examples=10,
        num_classes=5,
        base_seed=42
    )
    source2 = SemanticSegmentationRandomDataset(
        num_examples=20,  # Different dataset size
        num_classes=5,
        base_seed=42
    )

    dataset1 = PPSLDataset(source=source1, dataset_size=10)
    dataset2 = PPSLDataset(source=source2, dataset_size=20)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 != hash2, f"Different source datasets should produce different hashes: {hash1} == {hash2}"


def test_ppsl_different_source_seed_different_hash():
    """Test that different source seeds produce different hashes."""

    source1 = SemanticSegmentationRandomDataset(
        num_examples=10,
        num_classes=5,
        base_seed=42
    )
    source2 = SemanticSegmentationRandomDataset(
        num_examples=10,
        num_classes=5,
        base_seed=123  # Different seed
    )

    dataset1 = PPSLDataset(source=source1, dataset_size=10)
    dataset2 = PPSLDataset(source=source2, dataset_size=10)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 != hash2, f"Different source seeds should produce different hashes: {hash1} == {hash2}"


def test_ppsl_different_source_num_classes_different_hash():
    """Test that different source num_classes produce different hashes."""

    source1 = SemanticSegmentationRandomDataset(
        num_examples=10,
        num_classes=5,
        base_seed=42
    )
    source2 = SemanticSegmentationRandomDataset(
        num_examples=10,
        num_classes=10,  # Different number of classes
        base_seed=42
    )

    dataset1 = PPSLDataset(source=source1, dataset_size=10)
    dataset2 = PPSLDataset(source=source2, dataset_size=10)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 != hash2, f"Different source num_classes should produce different hashes: {hash1} == {hash2}"


def test_ppsl_hash_format():
    """Test that hash is in correct format."""

    source = SemanticSegmentationRandomDataset(
        num_examples=10,
        num_classes=5,
        base_seed=42
    )

    dataset = PPSLDataset(source=source, dataset_size=10)
    hash_val = dataset.get_cache_version_hash()

    # Should be a string
    assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"

    # Should be 16 characters (xxhash format)
    assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"

    # Should be hexadecimal
    assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"


def test_ppsl_comprehensive_no_hash_collisions():
    """Test that different configurations produce unique hashes (no collisions)."""

    # Create different source dataset configurations
    source_configs = [
        {'num_examples': 10, 'num_classes': 5, 'base_seed': 42},
        {'num_examples': 20, 'num_classes': 5, 'base_seed': 42},
        {'num_examples': 10, 'num_classes': 10, 'base_seed': 42},
        {'num_examples': 10, 'num_classes': 5, 'base_seed': 123},
        {'num_examples': 15, 'num_classes': 8, 'base_seed': 456},
    ]

    hashes = []
    for i, source_config in enumerate(source_configs):
        source = SemanticSegmentationRandomDataset(
            **source_config
        )

        dataset = PPSLDataset(source=source, dataset_size=source.num_examples)
        hash_val = dataset.get_cache_version_hash()

        # Check for collision
        assert hash_val not in hashes, f"Hash collision detected for config {i}: {source_config}, hash {hash_val} already exists"
        hashes.append(hash_val)

    # Verify we generated the expected number of unique hashes
    assert len(hashes) == len(source_configs), f"Expected {len(source_configs)} unique hashes, got {len(hashes)}"
