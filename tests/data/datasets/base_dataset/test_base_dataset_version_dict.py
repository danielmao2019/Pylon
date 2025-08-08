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
        
        # Both should have same class_name (data_root intentionally excluded from version dict)
        assert version_dict1['class_name'] == version_dict2['class_name']
