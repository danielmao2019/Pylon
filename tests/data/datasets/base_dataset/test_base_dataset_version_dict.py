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
        required_keys = {'class_name', 'data_root', 'split'}
        assert all(key in version_dict for key in required_keys), \
            f"Missing required keys: {required_keys - set(version_dict.keys())}"
        
        # Ensure data_root is absolute path
        assert temp_dir in version_dict['data_root'], \
            f"data_root must contain temp_dir, got {version_dict['data_root']}"
        
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



def test_version_dict_with_different_parameters(mock_dataset_class):
    """Test version dict structure with different parameter combinations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with string split
        dataset1 = mock_dataset_class(data_root=temp_dir, split='train')
        version_dict1 = dataset1._get_cache_version_dict()
        assert version_dict1['split'] == 'train'
        
        # Test with tuple split (split_percentages)
        dataset2 = mock_dataset_class(data_root=temp_dir, split=(0.7, 0.2, 0.1))
        version_dict2 = dataset2._get_cache_version_dict()
        assert version_dict2['split'] == (0.7, 0.2, 0.1)
        
        # Both should have same class_name and data_root
        assert version_dict1['class_name'] == version_dict2['class_name']
        assert version_dict1['data_root'] == version_dict2['data_root']