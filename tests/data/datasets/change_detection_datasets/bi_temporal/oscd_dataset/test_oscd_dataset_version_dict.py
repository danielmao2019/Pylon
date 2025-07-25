"""Tests for OSCDDataset version dict functionality."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset


def test_oscd_dataset_has_version_dict_method(oscd_dataset):
    """Test that OSCDDataset has _get_cache_version_dict method."""
    # Method should exist
    assert hasattr(oscd_dataset, '_get_cache_version_dict')
    assert callable(getattr(oscd_dataset, '_get_cache_version_dict'))


def test_oscd_dataset_version_dict_structure(oscd_dataset):
    """Test the structure and content of _get_cache_version_dict output."""
    version_dict = oscd_dataset._get_cache_version_dict()
    
    # Should return a dictionary
    assert isinstance(version_dict, dict)
    
    # Should contain base dataset parameters
    assert 'class_name' in version_dict
    assert version_dict['class_name'] == 'OSCDDataset'
    assert 'data_root' in version_dict
    assert 'split' in version_dict
    assert version_dict['split'] == 'train'


def test_oscd_dataset_version_dict_consistency(oscd_dataset):
    """Test that _get_cache_version_dict returns consistent results."""
    # Multiple calls should return the same result
    version_dict1 = oscd_dataset._get_cache_version_dict()
    version_dict2 = oscd_dataset._get_cache_version_dict()
    
    assert version_dict1 == version_dict2
