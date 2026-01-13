"""Test version hash discrimination for I3PEDataset.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
from data.datasets.change_detection_datasets.single_temporal.i3pe_dataset import I3PEDataset
from data.datasets.random_datasets.classification_random_dataset import ClassificationRandomDataset


def test_i3pe_same_parameters_same_hash():
    """Test that identical parameters produce identical hashes."""

    # Create identical source datasets
    source1a = ClassificationRandomDataset(
        num_examples=10,
        num_classes=5,
        image_res=(64, 64),
        base_seed=42
    )
    source1b = ClassificationRandomDataset(
        num_examples=10,
        num_classes=5,
        image_res=(64, 64),
        base_seed=42
    )

    # Same parameters should produce same hash
    dataset1a = I3PEDataset(source=source1a, dataset_size=10, exchange_ratio=0.75)
    dataset1b = I3PEDataset(source=source1b, dataset_size=10, exchange_ratio=0.75)

    hash1a = dataset1a.get_cache_version_hash()
    hash1b = dataset1b.get_cache_version_hash()

    assert hash1a == hash1b, f"Same parameters should produce same hash: {hash1a} != {hash1b}"


def test_i3pe_different_exchange_ratio_different_hash():
    """Test that different exchange_ratio values produce different hashes."""

    source = ClassificationRandomDataset(
        num_examples=10,
        num_classes=5,
        image_res=(64, 64),
        base_seed=42
    )

    dataset_075 = I3PEDataset(source=source, dataset_size=10, exchange_ratio=0.75)
    dataset_050 = I3PEDataset(source=source, dataset_size=10, exchange_ratio=0.50)

    hash_075 = dataset_075.get_cache_version_hash()
    hash_050 = dataset_050.get_cache_version_hash()

    assert hash_075 != hash_050, f"Different exchange_ratio should produce different hashes: {hash_075} == {hash_050}"


def test_i3pe_different_source_different_hash():
    """Test that different source datasets produce different hashes."""

    # Create different source datasets
    source1 = ClassificationRandomDataset(
        num_examples=10,
        num_classes=5,
        image_res=(64, 64),
        base_seed=42
    )
    source2 = ClassificationRandomDataset(
        num_examples=20,  # Different dataset size
        num_classes=5,
        image_res=(64, 64),
        base_seed=42
    )

    dataset1 = I3PEDataset(source=source1, dataset_size=10, exchange_ratio=0.75)
    dataset2 = I3PEDataset(source=source2, dataset_size=20, exchange_ratio=0.75)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 != hash2, f"Different source datasets should produce different hashes: {hash1} == {hash2}"


def test_i3pe_different_source_seed_different_hash():
    """Test that different source seeds produce different hashes."""

    source1 = ClassificationRandomDataset(
        num_examples=10,
        num_classes=5,
        image_res=(64, 64),
        base_seed=42
    )
    source2 = ClassificationRandomDataset(
        num_examples=10,
        num_classes=5,
        image_res=(64, 64),
        base_seed=123  # Different seed
    )

    dataset1 = I3PEDataset(source=source1, dataset_size=10, exchange_ratio=0.75)
    dataset2 = I3PEDataset(source=source2, dataset_size=20, exchange_ratio=0.75)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 != hash2, f"Different source seeds should produce different hashes: {hash1} == {hash2}"


def test_i3pe_hash_format():
    """Test that hash is in correct format."""

    source = ClassificationRandomDataset(
        num_examples=10,
        num_classes=5,
        image_res=(64, 64),
        base_seed=42
    )

    dataset = I3PEDataset(source=source, dataset_size=10, exchange_ratio=0.75)
    hash_val = dataset.get_cache_version_hash()

    # Should be a string
    assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"

    # Should be 16 characters (xxhash format)
    assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"

    # Should be hexadecimal
    assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"


def test_i3pe_comprehensive_no_hash_collisions():
    """Test that different configurations produce unique hashes (no collisions)."""

    # Create different source datasets and configurations
    configs = []

    # Different source configurations
    source_configs = [
        {'num_examples': 10, 'num_classes': 5, 'base_seed': 42},
        {'num_examples': 20, 'num_classes': 5, 'base_seed': 42},
        {'num_examples': 10, 'num_classes': 10, 'base_seed': 42},
        {'num_examples': 10, 'num_classes': 5, 'base_seed': 123},
    ]

    exchange_ratios = [0.50, 0.75]

    for source_config in source_configs:
        for exchange_ratio in exchange_ratios:
            source = ClassificationRandomDataset(
                image_res=(64, 64),
                **source_config
            )
            configs.append({
                'source': source,
                'exchange_ratio': exchange_ratio,
                'config_desc': f"{source_config}_ratio_{exchange_ratio}"
            })

    hashes = []
    for config in configs:
        dataset = I3PEDataset(source=config['source'], dataset_size=config['source'].num_examples, exchange_ratio=config['exchange_ratio'])
        hash_val = dataset.get_cache_version_hash()

        # Check for collision
        assert hash_val not in hashes, f"Hash collision detected for config {config['config_desc']}: hash {hash_val} already exists"
        hashes.append(hash_val)

    # Verify we generated the expected number of unique hashes
    assert len(hashes) == len(configs), f"Expected {len(configs)} unique hashes, got {len(hashes)}"
