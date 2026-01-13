"""Test version hash discrimination for Urb3DCDDataset.

Focus: Ensure different dataset configurations produce different cache version hashes.
"""

import pytest
import tempfile
import os
from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset



def test_urb3dcd_same_parameters_same_hash(urb3dcd_data_root):
    """Test that identical parameters produce identical hashes."""
    # Same parameters should produce same hash
    dataset1a = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', version=1, patched=True)
    dataset1b = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', version=1, patched=True)

    hash1a = dataset1a.get_cache_version_hash()
    hash1b = dataset1b.get_cache_version_hash()

    assert hash1a == hash1b, f"Same parameters should produce same hash: {hash1a} != {hash1b}"


def test_urb3dcd_different_split_different_hash(urb3dcd_data_root):
    """Test that different splits produce different hashes."""
    dataset_train = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', version=1)
    dataset_val = Urb3DCDDataset(data_root=urb3dcd_data_root, split='val', version=1)
    dataset_test = Urb3DCDDataset(data_root=urb3dcd_data_root, split='test', version=1)

    hash_train = dataset_train.get_cache_version_hash()
    hash_val = dataset_val.get_cache_version_hash()
    hash_test = dataset_test.get_cache_version_hash()

    assert hash_train != hash_val, f"Different splits should produce different hashes: {hash_train} == {hash_val}"
    assert hash_train != hash_test, f"Different splits should produce different hashes: {hash_train} == {hash_test}"
    assert hash_val != hash_test, f"Different splits should produce different hashes: {hash_val} == {hash_test}"


def test_urb3dcd_different_version_different_hash(urb3dcd_data_root):
    """Test that different version values produce different hashes."""
    dataset_v1 = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', version=1)
    dataset_v2 = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', version=2)

    hash_v1 = dataset_v1.get_cache_version_hash()
    hash_v2 = dataset_v2.get_cache_version_hash()

    assert hash_v1 != hash_v2, f"Different versions should produce different hashes: {hash_v1} == {hash_v2}"


def test_urb3dcd_different_patched_different_hash(urb3dcd_data_root):
    """Test that different patched values produce different hashes."""
    dataset_patched = Urb3DCDDataset(data_root=urb3dcd_data_root, split='val', patched=True)
    dataset_not_patched = Urb3DCDDataset(data_root=urb3dcd_data_root, split='val', patched=False)

    hash_patched = dataset_patched.get_cache_version_hash()
    hash_not_patched = dataset_not_patched.get_cache_version_hash()

    assert hash_patched != hash_not_patched, f"Different patched values should produce different hashes: {hash_patched} == {hash_not_patched}"


def test_urb3dcd_different_sample_per_epoch_different_hash(urb3dcd_data_root):
    """Test that different sample_per_epoch values produce different hashes."""
    dataset_64 = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', sample_per_epoch=64)
    dataset_128 = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', sample_per_epoch=128)

    hash_64 = dataset_64.get_cache_version_hash()
    hash_128 = dataset_128.get_cache_version_hash()

    assert hash_64 != hash_128, f"Different sample_per_epoch should produce different hashes: {hash_64} == {hash_128}"


def test_urb3dcd_different_fix_samples_different_hash(urb3dcd_data_root):
    """Test that different fix_samples values produce different hashes."""
    dataset_fixed = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', fix_samples=True)
    dataset_not_fixed = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', fix_samples=False)

    hash_fixed = dataset_fixed.get_cache_version_hash()
    hash_not_fixed = dataset_not_fixed.get_cache_version_hash()

    assert hash_fixed != hash_not_fixed, f"Different fix_samples should produce different hashes: {hash_fixed} == {hash_not_fixed}"


def test_urb3dcd_different_radius_different_hash(urb3dcd_data_root):
    """Test that different radius values produce different hashes."""
    dataset_25 = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', radius=25.0)
    dataset_50 = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', radius=50.0)

    hash_25 = dataset_25.get_cache_version_hash()
    hash_50 = dataset_50.get_cache_version_hash()

    assert hash_25 != hash_50, f"Different radius should produce different hashes: {hash_25} == {hash_50}"


def test_urb3dcd_same_data_root_same_hash(urb3dcd_data_root):
    """Test that same data roots produce same hashes (data_root excluded from hash)."""
    # data_root is intentionally excluded from hash for cache stability
    dataset1 = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', version=1)
    dataset2 = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', version=1)

    hash1 = dataset1.get_cache_version_hash()
    hash2 = dataset2.get_cache_version_hash()

    assert hash1 == hash2, f"Same data roots should produce same hashes: {hash1} != {hash2}"


def test_urb3dcd_hash_format(urb3dcd_data_root):
    """Test that hash is in correct format."""
    dataset = Urb3DCDDataset(data_root=urb3dcd_data_root, split='train', version=1)
    hash_val = dataset.get_cache_version_hash()

    # Should be a string
    assert isinstance(hash_val, str), f"Hash should be string, got {type(hash_val)}"

    # Should be 16 characters (xxhash format)
    assert len(hash_val) == 16, f"Hash should be 16 characters, got {len(hash_val)}"

    # Should be hexadecimal
    assert all(c in '0123456789abcdef' for c in hash_val.lower()), f"Hash should be hexadecimal: {hash_val}"


def test_urb3dcd_comprehensive_no_hash_collisions(urb3dcd_data_root):
    """Test that different configurations produce unique hashes (no collisions)."""
    # Test various parameter combinations
    # NOTE: data_root is intentionally excluded from hash, so we test only meaningful parameter combinations
    configs = [
        {'data_root': urb3dcd_data_root, 'split': 'train', 'version': 1, 'patched': True, 'sample_per_epoch': 64},
        {'data_root': urb3dcd_data_root, 'split': 'train', 'version': 1, 'patched': True, 'sample_per_epoch': 128},
        {'data_root': urb3dcd_data_root, 'split': 'train', 'version': 1, 'patched': False},
        {'data_root': urb3dcd_data_root, 'split': 'val', 'version': 1, 'patched': True},
        {'data_root': urb3dcd_data_root, 'split': 'test', 'version': 1, 'patched': False},
        {'data_root': urb3dcd_data_root, 'split': 'train', 'version': 1, 'fix_samples': True},
        {'data_root': urb3dcd_data_root, 'split': 'train', 'version': 1, 'radius': 25.0},
        # Removed different data_root configs since data_root is excluded from hash
    ]

    hashes = []
    for config in configs:
        dataset = Urb3DCDDataset(**config)
        hash_val = dataset.get_cache_version_hash()

        # Check for collision
        assert hash_val not in hashes, f"Hash collision detected for config {config}: hash {hash_val} already exists"
        hashes.append(hash_val)

    # Verify we generated the expected number of unique hashes
    assert len(hashes) == len(configs), f"Expected {len(configs)} unique hashes, got {len(hashes)}"
