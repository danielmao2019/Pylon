"""Tests for GANDataset cache version discrimination."""

import pytest
import tempfile
import torch
from data.datasets.gan_datasets.gan_dataset import GANDataset
from data.datasets.torchvision_datasets.mnist import MNISTDataset


def test_gan_dataset_version_discrimination():
    """Test that GANDataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a source dataset
        source_dataset = MNISTDataset(
            data_root=temp_dir,
            split='train'
        )

        # Same parameters should have same hash
        dataset1a = GANDataset(
            source=source_dataset,
            latent_dim=100
        )
        dataset1b = GANDataset(
            source=source_dataset,
            latent_dim=100
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()

        # Different latent_dim should have different hash
        dataset2 = GANDataset(
            source=source_dataset,
            latent_dim=200  # Different
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()

        # Different source dataset should have different hash
        source_dataset2 = MNISTDataset(
            data_root=temp_dir,
            split='test'  # Different split
        )
        dataset3 = GANDataset(
            source=source_dataset2,  # Different source
            latent_dim=100
        )
        assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()


def test_latent_dim_variants():
    """Test that different latent dimensions produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dataset = MNISTDataset(
            data_root=temp_dir,
            split='train'
        )

        latent_dim_variants = [50, 100, 200, 512]

        datasets = []
        for latent_dim in latent_dim_variants:
            dataset = GANDataset(
                source=source_dataset,
                latent_dim=latent_dim
            )
            datasets.append(dataset)

        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All latent_dim variants should produce different hashes, got: {hashes}"


def test_different_source_datasets():
    """Test that different source datasets produce different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create different source datasets
        source1 = MNISTDataset(data_root=temp_dir, split='train')
        source2 = MNISTDataset(data_root=temp_dir, split='test')

        dataset1 = GANDataset(source=source1, latent_dim=100)
        dataset2 = GANDataset(source=source2, latent_dim=100)

        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_inherited_parameters_affect_version_hash():
    """Test that parameters that should affect dataset content produce different hashes.

    Note: Only parameters that affect the actual data content should affect the cache
    version hash. Implementation details like caching behavior and device placement
    should not affect the hash as they don't change the data.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        source_dataset = MNISTDataset(
            data_root=temp_dir,
            split='train'
        )

        # Test different dataset_size parameter which affects the dataset content
        dataset1 = GANDataset(source=source_dataset, latent_dim=100)
        dataset2 = GANDataset(source=source_dataset, latent_dim=100, dataset_size=50)  # Different size

        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
            "Different dataset_size should affect cache version hash"


def test_source_dataset_parameters_propagation():
    """Test that changes in source dataset parameters affect GANDataset hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create source datasets with different splits (which should affect the hash)
        source1 = MNISTDataset(data_root=temp_dir, split='train')
        source2 = MNISTDataset(data_root=temp_dir, split='test')

        dataset1 = GANDataset(source=source1, latent_dim=100)
        dataset2 = GANDataset(source=source2, latent_dim=100)

        # Should have different hashes due to different source dataset parameters
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        datasets = []

        # Create different source datasets
        source_train = MNISTDataset(data_root=temp_dir, split='train')
        source_test = MNISTDataset(data_root=temp_dir, split='test')

        sources = [source_train, source_test]

        # Generate many different dataset configurations
        for source in sources:
            for latent_dim in [50, 100, 200]:
                for dataset_size in [None, 100]:  # Test different dataset sizes
                    datasets.append(GANDataset(
                        source=source,
                        latent_dim=latent_dim,
                        dataset_size=dataset_size
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


