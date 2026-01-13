"""CelebA-specific hash discrimination tests.

This module contains tests specific to CelebA dataset parameters that don't
apply to other datasets, particularly the use_landmarks parameter.
"""

import pytest
from utils.builders.builder import build_from_config


def test_celeb_a_use_landmarks_discrimination(celeb_a_data_root):
    """Test CelebA-specific parameter discrimination (use_landmarks)."""
    celeb_a_config = {
        'class': None,  # Will be imported below
        'args': {
            'data_root': celeb_a_data_root,
            'split': 'train'
        }
    }

    # Import here to avoid import issues
    from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset
    celeb_a_config['class'] = CelebADataset

    # Test that use_landmarks=True vs use_landmarks=False produce different hashes
    config_with_landmarks = {
        'class': CelebADataset,
        'args': {**celeb_a_config['args'], 'use_landmarks': True}
    }

    config_without_landmarks = {
        'class': CelebADataset,
        'args': {**celeb_a_config['args'], 'use_landmarks': False}
    }

    dataset_with_landmarks = build_from_config(config_with_landmarks)
    dataset_without_landmarks = build_from_config(config_without_landmarks)

    hash_with = dataset_with_landmarks.get_cache_version_hash()
    hash_without = dataset_without_landmarks.get_cache_version_hash()

    assert hash_with != hash_without, f"Different use_landmarks should produce different hashes: {hash_with} == {hash_without}"


def test_celeb_a_landmarks_split_combinations(celeb_a_data_root):
    """Test combinations of landmarks and splits produce unique hashes."""
    from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset

    base_config = {
        'data_root': celeb_a_data_root,
    }

    test_combinations = [
        {'split': 'train', 'use_landmarks': True},
        {'split': 'train', 'use_landmarks': False},
        {'split': 'val', 'use_landmarks': True},
        {'split': 'val', 'use_landmarks': False},
    ]

    hashes = []
    for combination in test_combinations:
        config = {
            'class': CelebADataset,
            'args': {**base_config, **combination}
        }
        dataset = build_from_config(config)
        hash_val = dataset.get_cache_version_hash()

        # Check for collision
        assert hash_val not in hashes, f"Hash collision detected for {combination}: hash {hash_val} already exists"
        hashes.append(hash_val)

    # Verify we have 4 unique hashes
    assert len(hashes) == 4, f"Expected 4 unique hashes, got {len(hashes)}"