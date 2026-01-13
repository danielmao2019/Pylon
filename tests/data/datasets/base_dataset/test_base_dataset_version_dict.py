"""Tests for BaseDataset version dict functionality."""

import pytest
import tempfile


def test_base_dataset_has_version_dict_method(mock_dataset_class):
    """Test that BaseDataset has _get_cache_version_dict method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = mock_dataset_class(data_root=temp_dir, split='train')

        # Method should exist
        assert hasattr(dataset, '_get_cache_version_dict')
        assert callable(getattr(dataset, '_get_cache_version_dict'))


def test_base_dataset_version_dict_structure(mock_dataset_class):
    """Test the structure and content of _get_cache_version_dict output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = mock_dataset_class(data_root=temp_dir, split='train')
        version_dict = dataset._get_cache_version_dict()

        # Should return a dictionary
        assert isinstance(version_dict, dict)

        # Ensure essential keys are present
        # NOTE: data_root is intentionally excluded for cache stability across different paths
        required_keys = {'class_name', 'split'}
        assert all(key in version_dict for key in required_keys), \
            f"Missing required keys: {required_keys - set(version_dict.keys())}"

        # Ensure class_name matches actual class
        assert version_dict['class_name'] == dataset.__class__.__name__


def test_base_dataset_version_dict_consistency(mock_dataset_class):
    """Test that _get_cache_version_dict returns consistent results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = mock_dataset_class(data_root=temp_dir, split='train')

        # Multiple calls should return the same result
        version_dict1 = dataset._get_cache_version_dict()
        version_dict2 = dataset._get_cache_version_dict()

        assert version_dict1 == version_dict2



def test_version_dict_with_different_parameters(mock_dataset_class, mock_dataset_class_without_predefined_splits):
    """Test version dict structure with different parameter combinations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with string split
        dataset1 = mock_dataset_class(data_root=temp_dir, split='train')
        version_dict1 = dataset1._get_cache_version_dict()
        assert version_dict1['split'] == 'train'

        # Test with split_percentages (use dataset without predefined splits)
        dataset2 = mock_dataset_class_without_predefined_splits(data_root=temp_dir, split='train', split_percentages=(0.7, 0.2, 0.1))
        version_dict2 = dataset2._get_cache_version_dict()
        assert version_dict2['split'] == 'train'
        assert version_dict2['split_percentages'] == (0.7, 0.2, 0.1)

        # Different classes should have different class_names
        assert version_dict1['class_name'] != version_dict2['class_name']
        assert version_dict1['class_name'] == 'MockDataset'
        assert version_dict2['class_name'] == 'MockDatasetWithoutPredefinedSplits'


def test_version_dict_comprehensive_scenarios(mock_dataset_class, mock_dataset_class_without_predefined_splits):
    """Comprehensive test for all version dict scenarios and parameter combinations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Scenario 1: Predefined split only
        dataset1 = mock_dataset_class(data_root=temp_dir, split='train')
        version_dict1 = dataset1._get_cache_version_dict()

        # Should have split but not split_percentages
        assert 'split' in version_dict1
        assert version_dict1['split'] == 'train'
        assert 'split_percentages' not in version_dict1
        assert 'class_name' in version_dict1
        assert 'base_seed' in version_dict1  # Should have default base_seed

        # Scenario 2: Split with percentages
        dataset2 = mock_dataset_class_without_predefined_splits(
            data_root=temp_dir,
            split='train',
            split_percentages=(0.7, 0.2, 0.1),
            base_seed=42
        )
        version_dict2 = dataset2._get_cache_version_dict()

        # Should have both split and split_percentages
        assert 'split' in version_dict2
        assert version_dict2['split'] == 'train'
        assert 'split_percentages' in version_dict2
        assert version_dict2['split_percentages'] == (0.7, 0.2, 0.1)
        assert 'base_seed' in version_dict2
        assert version_dict2['base_seed'] == 42

        # Scenario 3: Split=None
        dataset3 = mock_dataset_class_without_predefined_splits(data_root=temp_dir, split=None)
        version_dict3 = dataset3._get_cache_version_dict()

        # Should NOT have split or split_percentages
        assert 'split' not in version_dict3
        assert 'split_percentages' not in version_dict3
        assert 'class_name' in version_dict3
        assert 'base_seed' in version_dict3  # Should have default base_seed

        # Scenario 4: Split with custom indices
        dataset4 = mock_dataset_class(data_root=temp_dir, split='train', indices=[1, 3, 5])
        version_dict4 = dataset4._get_cache_version_dict()

        # Should have split and indices
        assert 'split' in version_dict4
        assert version_dict4['split'] == 'train'
        assert 'indices' in version_dict4
        assert version_dict4['indices'] == [1, 3, 5]
        assert 'split_percentages' not in version_dict4

        # Scenario 5: Split percentages with indices
        dataset5 = mock_dataset_class_without_predefined_splits(
            data_root=temp_dir,
            split='val',
            split_percentages=(0.8, 0.1, 0.1),
            indices=[0, 2],
            base_seed=123
        )
        version_dict5 = dataset5._get_cache_version_dict()

        # Should have split, split_percentages, and indices
        assert 'split' in version_dict5
        assert version_dict5['split'] == 'val'
        assert 'split_percentages' in version_dict5
        assert version_dict5['split_percentages'] == (0.8, 0.1, 0.1)
        assert 'indices' in version_dict5
        assert version_dict5['indices'] == [0, 2]
        assert 'base_seed' in version_dict5
        assert version_dict5['base_seed'] == 123

        # Verify all version dicts are unique
        version_dicts = [version_dict1, version_dict2, version_dict3, version_dict4, version_dict5]
        hashes = [dataset.get_cache_version_hash() for dataset in [dataset1, dataset2, dataset3, dataset4, dataset5]]
        assert len(set(hashes)) == len(hashes), f"Version hashes should be unique, got: {hashes}"

        # Verify data_root is excluded (cache stability)
        for version_dict in version_dicts:
            assert 'data_root' not in version_dict, f"data_root should be excluded from version dict: {version_dict}"
