"""Tests for OSCDDataset version dict functionality."""

import pytest
import tempfile
import os
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset




def test_oscd_dataset_has_version_dict_method(create_dummy_oscd_files):
    """Test that OSCDDataset has _get_cache_version_dict method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_oscd_files(temp_dir)
        
        dataset = OSCDDataset(data_root=temp_dir, split='train')
        
        # Method should exist
        assert hasattr(dataset, '_get_cache_version_dict')
        assert callable(getattr(dataset, '_get_cache_version_dict'))


def test_oscd_dataset_version_dict_structure(create_dummy_oscd_files):
    """Test the structure and content of _get_cache_version_dict output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_oscd_files(temp_dir)
        
        dataset = OSCDDataset(data_root=temp_dir, split='train')
        version_dict = dataset._get_cache_version_dict()
        
        # Should return a dictionary
        assert isinstance(version_dict, dict)
        
        # Should contain base dataset parameters
        assert 'class_name' in version_dict
        assert version_dict['class_name'] == 'OSCDDataset'
        assert 'data_root' in version_dict
        assert 'split' in version_dict
        assert version_dict['split'] == 'train'


def test_oscd_dataset_version_dict_consistency(create_dummy_oscd_files):
    """Test that _get_cache_version_dict returns consistent results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_oscd_files(temp_dir)
        
        dataset = OSCDDataset(data_root=temp_dir, split='train')
        
        # Multiple calls should return the same result
        version_dict1 = dataset._get_cache_version_dict()
        version_dict2 = dataset._get_cache_version_dict()
        
        assert version_dict1 == version_dict2


