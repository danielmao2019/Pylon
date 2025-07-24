"""Tests for KITTIDataset version dict functionality."""

import pytest
import tempfile
import os
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset




def test_kitti_dataset_has_version_dict_method(create_dummy_kitti_structure, patched_kitti_dataset_size):
    """Test that KITTIDataset has _get_cache_version_dict method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kitti_structure(temp_dir)
        with patched_kitti_dataset_size():
            dataset = KITTIDataset(data_root=temp_dir, split='train')
        
        # Method should exist
        assert hasattr(dataset, '_get_cache_version_dict')
        assert callable(getattr(dataset, '_get_cache_version_dict'))


def test_kitti_dataset_version_dict_structure(create_dummy_kitti_structure, patched_kitti_dataset_size):
    """Test the structure and content of _get_cache_version_dict output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kitti_structure(temp_dir)
        with patched_kitti_dataset_size():
            dataset = KITTIDataset(data_root=temp_dir, split='train')
        version_dict = dataset._get_cache_version_dict()
        
        # Should return a dictionary
        assert isinstance(version_dict, dict)
        
        # Should contain base dataset parameters
        assert 'class_name' in version_dict
        assert version_dict['class_name'] == 'KITTIDataset'
        assert 'data_root' in version_dict
        assert 'split' in version_dict
        assert version_dict['split'] == 'train'


def test_kitti_dataset_version_dict_consistency(create_dummy_kitti_structure, patched_kitti_dataset_size):
    """Test that _get_cache_version_dict returns consistent results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kitti_structure(temp_dir)
        with patched_kitti_dataset_size():
            dataset = KITTIDataset(data_root=temp_dir, split='train')
        
        # Multiple calls should return the same result
        version_dict1 = dataset._get_cache_version_dict()
        version_dict2 = dataset._get_cache_version_dict()
        
        assert version_dict1 == version_dict2


def test_kitti_dataset_get_cache_version_hash_method(create_dummy_kitti_structure, patched_kitti_dataset_size):
    """Test that KITTIDataset has get_cache_version_hash method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kitti_structure(temp_dir)
        with patched_kitti_dataset_size():
            dataset = KITTIDataset(data_root=temp_dir, split='train')
        
        # Method should exist and return a string
        assert hasattr(dataset, 'get_cache_version_hash')
        assert callable(getattr(dataset, 'get_cache_version_hash'))
        
        hash_val = dataset.get_cache_version_hash()
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16  # xxhash produces 16-character hex strings