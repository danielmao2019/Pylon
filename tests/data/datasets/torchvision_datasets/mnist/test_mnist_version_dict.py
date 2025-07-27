"""Tests for MNISTDataset version dict functionality."""

import pytest
import tempfile
from data.datasets.torchvision_datasets.mnist import MNISTDataset


def test_mnist_dataset_has_version_dict_method():
    """Test that MNISTDataset has _get_cache_version_dict method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = MNISTDataset(data_root=temp_dir, split='train')
        
        # Method should exist
        assert hasattr(dataset, '_get_cache_version_dict')
        assert callable(getattr(dataset, '_get_cache_version_dict'))


def test_mnist_dataset_version_dict_structure():
    """Test the structure and content of _get_cache_version_dict output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = MNISTDataset(data_root=temp_dir, split='train')
        version_dict = dataset._get_cache_version_dict()
        
        # Should return a dictionary
        assert isinstance(version_dict, dict)
        
        # Should contain base dataset parameters (data_root intentionally excluded for cache stability)
        assert 'class_name' in version_dict
        assert version_dict['class_name'] == 'MNISTDataset'
        assert 'split' in version_dict
        assert version_dict['split'] == 'train'


def test_mnist_dataset_version_dict_consistency():
    """Test that _get_cache_version_dict returns consistent results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = MNISTDataset(data_root=temp_dir, split='train')
        
        # Multiple calls should return the same result
        version_dict1 = dataset._get_cache_version_dict()
        version_dict2 = dataset._get_cache_version_dict()
        
        assert version_dict1 == version_dict2


