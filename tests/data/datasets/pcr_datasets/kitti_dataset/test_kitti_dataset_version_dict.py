"""Tests for KITTIDataset version dict functionality."""

import pytest
from utils.builders.builder import build_from_config




@pytest.mark.parametrize('kitti_dataset_config', ['train'], indirect=True)
def test_kitti_dataset_has_version_dict_method(kitti_dataset_config):
    """Test that KITTIDataset has _get_cache_version_dict method."""
    dataset = build_from_config(kitti_dataset_config)

    # Method should exist
    assert hasattr(dataset, '_get_cache_version_dict')
    assert callable(getattr(dataset, '_get_cache_version_dict'))


@pytest.mark.parametrize('kitti_dataset_config', ['train'], indirect=True)
def test_kitti_dataset_version_dict_structure(kitti_dataset_config):
    """Test the structure and content of _get_cache_version_dict output."""
    dataset = build_from_config(kitti_dataset_config)
    version_dict = dataset._get_cache_version_dict()

    # Should return a dictionary
    assert isinstance(version_dict, dict)

    # Should contain base dataset parameters (data_root intentionally excluded for cache stability)
    assert 'class_name' in version_dict
    assert version_dict['class_name'] == 'KITTIDataset'
    assert 'split' in version_dict
    assert version_dict['split'] == 'train'


@pytest.mark.parametrize('kitti_dataset_config', ['train'], indirect=True)
def test_kitti_dataset_version_dict_consistency(kitti_dataset_config):
    """Test that _get_cache_version_dict returns consistent results."""
    dataset = build_from_config(kitti_dataset_config)

    # Multiple calls should return the same result
    version_dict1 = dataset._get_cache_version_dict()
    version_dict2 = dataset._get_cache_version_dict()

    assert version_dict1 == version_dict2


@pytest.mark.parametrize('kitti_dataset_config', ['train'], indirect=True)
def test_kitti_dataset_get_cache_version_hash_method(kitti_dataset_config):
    """Test that KITTIDataset has get_cache_version_hash method."""
    dataset = build_from_config(kitti_dataset_config)

    # Method should exist and return a string
    assert hasattr(dataset, 'get_cache_version_hash')
    assert callable(getattr(dataset, 'get_cache_version_hash'))

    hash_val = dataset.get_cache_version_hash()
    assert isinstance(hash_val, str)
    assert len(hash_val) == 16  # xxhash produces 16-character hex strings
