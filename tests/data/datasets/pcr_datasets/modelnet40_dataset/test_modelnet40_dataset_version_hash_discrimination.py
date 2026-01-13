"""Tests for ModelNet40Dataset cache version discrimination."""

import pytest
import tempfile
import os
from data.datasets.pcr_datasets.modelnet40_dataset import ModelNet40Dataset


def create_dummy_modelnet40_structure(data_root: str) -> None:
    """Create a dummy ModelNet40 directory structure for testing."""
    categories = ['airplane', 'chair', 'table']  # Subset for testing
    splits = ['train', 'test']

    for category in categories:
        for split in splits:
            category_dir = os.path.join(data_root, category, split)
            os.makedirs(category_dir, exist_ok=True)

            # Create a few dummy OFF files
            for i in range(2):
                off_file = os.path.join(category_dir, f'{category}_{i:04d}.off')
                with open(off_file, 'w') as f:
                    # Dummy OFF content
                    f.write('OFF\n3 1 0\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n3 0 1 2\n')


def test_modelnet40_dataset_version_discrimination():
    """Test that ModelNet40Dataset instances with different parameters have different hashes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_modelnet40_structure(temp_dir)

        # Same parameters should have same hash
        dataset1a = ModelNet40Dataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05,
            split='train'
        )
        dataset1b = ModelNet40Dataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05,
            split='train'
        )
        assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()

        # Different dataset_size should have different hash
        dataset2 = ModelNet40Dataset(
            data_root=temp_dir,
            dataset_size=200,  # Different
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05,
            split='train'
        )
        assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()

        # Different rotation_mag should have different hash
        dataset3 = ModelNet40Dataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=30.0,  # Different
            translation_mag=0.5,
            matching_radius=0.05,
            split='train'
        )
        assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()

        # Different split should have different hash
        dataset4 = ModelNet40Dataset(
            data_root=temp_dir,
            dataset_size=100,
            rotation_mag=45.0,
            translation_mag=0.5,
            matching_radius=0.05,
            split='test'  # Different
        )
        assert dataset1a.get_cache_version_hash() != dataset4.get_cache_version_hash()


def test_inherited_parameters_affect_version_hash():
    """Test that parameters inherited from SyntheticTransformPCRDataset affect version hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_modelnet40_structure(temp_dir)

        base_args = {
            'data_root': temp_dir,
            'dataset_size': 100,
            'split': 'train',
        }

        # Test inherited parameters from SyntheticTransformPCRDataset
        parameter_variants = [
            ('rotation_mag', 30.0),  # Different from default 45.0
            ('translation_mag', 0.3),  # Different from default 0.5
            ('matching_radius', 0.1),  # Different from default 0.05
            ('overlap_range', (0.2, 0.9)),  # Different from default (0.3, 1.0)
            ('min_points', 256),  # Different from default 512
            ('max_trials', 500),  # Different from default 1000
        ]

        dataset1 = ModelNet40Dataset(**base_args)

        for param_name, new_value in parameter_variants:
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = ModelNet40Dataset(**modified_args)

            assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
                f"Inherited parameter {param_name} should affect cache version hash"


def test_modelnet40_specific_parameters():
    """Test that ModelNet40-specific features affect version hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create different ModelNet40 structures to test category effects
        categories_set1 = ['airplane', 'chair']
        categories_set2 = ['airplane', 'chair', 'table']  # Additional category

        # Structure 1
        for category in categories_set1:
            for split in ['train', 'test']:
                category_dir = os.path.join(temp_dir, 'structure1', category, split)
                os.makedirs(category_dir, exist_ok=True)
                off_file = os.path.join(category_dir, f'{category}_0001.off')
                with open(off_file, 'w') as f:
                    f.write('OFF\n3 1 0\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n3 0 1 2\n')

        # Structure 2 - with additional category
        for category in categories_set2:
            for split in ['train', 'test']:
                category_dir = os.path.join(temp_dir, 'structure2', category, split)
                os.makedirs(category_dir, exist_ok=True)
                off_file = os.path.join(category_dir, f'{category}_0001.off')
                with open(off_file, 'w') as f:
                    f.write('OFF\n3 1 0\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n3 0 1 2\n')

        # Same parameters but different data_root (different available categories)
        dataset1 = ModelNet40Dataset(
            data_root=os.path.join(temp_dir, 'structure1'),
            dataset_size=100,
            split='train'
        )
        dataset2 = ModelNet40Dataset(
            data_root=os.path.join(temp_dir, 'structure2'),
            dataset_size=100,
            split='train'
        )

        # Should have different hashes due to different available categories affecting file_pair_annotations
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()


def test_keep_ratio_variants():
    """Test that different keep_ratio values are properly discriminated."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_modelnet40_structure(temp_dir)

        keep_ratio_variants = [0.5, 0.7, 0.8, 0.9]

        datasets = []
        for keep_ratio in keep_ratio_variants:
            dataset = ModelNet40Dataset(
                data_root=temp_dir,
                dataset_size=100,
                split='train',
                keep_ratio=keep_ratio
            )
            datasets.append(dataset)

        # All should have different hashes
        hashes = [dataset.get_cache_version_hash() for dataset in datasets]
        assert len(hashes) == len(set(hashes)), \
            f"All keep_ratio variants should produce different hashes, got: {hashes}"


def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_modelnet40_structure(temp_dir)

        datasets = []

        # Generate many different dataset configurations
        for dataset_size in [50, 100, 200]:
            for rotation_mag in [30.0, 45.0, 60.0]:
                for translation_mag in [0.3, 0.5, 0.7]:
                    for split in ['train', 'test']:
                        datasets.append(ModelNet40Dataset(
                            data_root=temp_dir,
                            dataset_size=dataset_size,
                            rotation_mag=rotation_mag,
                            translation_mag=translation_mag,
                            split=split
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


