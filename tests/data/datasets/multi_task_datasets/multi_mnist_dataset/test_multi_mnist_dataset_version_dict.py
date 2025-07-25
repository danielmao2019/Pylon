"""Tests for MultiMNISTDataset version dict functionality."""

import pytest
import tempfile
from data.datasets.multi_task_datasets.multi_mnist_dataset import MultiMNISTDataset


def test_multi_mnist_dataset_has_version_dict_method():
    """Test that MultiMNISTDataset has _get_cache_version_dict method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = MultiMNISTDataset(data_root=temp_dir, split='train')
        
        # Method should exist
        assert hasattr(dataset, '_get_cache_version_dict')
        assert callable(getattr(dataset, '_get_cache_version_dict'))


def test_multi_mnist_dataset_version_dict_structure():
    """Test the structure and content of _get_cache_version_dict output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = MultiMNISTDataset(data_root=temp_dir, split='train')
        version_dict = dataset._get_cache_version_dict()
        
        # Should return a dictionary
        assert isinstance(version_dict, dict)
        
        # Should contain base dataset parameters
        assert 'class_name' in version_dict
        assert version_dict['class_name'] == 'MultiMNISTDataset'
        assert 'data_root' in version_dict
        assert 'split' in version_dict
        assert version_dict['split'] == 'train'


def test_multi_mnist_dataset_version_dict_consistency():
    """Test that _get_cache_version_dict returns consistent results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = MultiMNISTDataset(data_root=temp_dir, split='train')
        
        # Multiple calls should return the same result
        version_dict1 = dataset._get_cache_version_dict()
        version_dict2 = dataset._get_cache_version_dict()
        
        assert version_dict1 == version_dict2


def test_multi_mnist_dataset_get_cache_version_hash_method():
    """Test that MultiMNISTDataset has get_cache_version_hash method."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = MultiMNISTDataset(data_root=temp_dir, split='train')
        
        # Method should exist and return a string
        assert hasattr(dataset, 'get_cache_version_hash')
        assert callable(getattr(dataset, 'get_cache_version_hash'))
        
        hash_val = dataset.get_cache_version_hash()
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16  # xxhash produces 16-character hex strings