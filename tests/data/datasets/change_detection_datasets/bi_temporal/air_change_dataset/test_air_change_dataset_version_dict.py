"""Test version dict implementation for AirChangeDataset."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset


def test_air_change_dataset_has_version_dict_method():
    """Test that AirChangeDataset has _get_cache_version_dict method."""
    assert hasattr(AirChangeDataset, '_get_cache_version_dict')
    
    # Check method signature
    import inspect
    method = getattr(AirChangeDataset, '_get_cache_version_dict')
    signature = inspect.signature(method)
    
    # Should take only self parameter
    params = list(signature.parameters.keys())
    assert params == ['self']
    
    # Should return Dict[str, Any]
    from typing import Dict, Any
    return_annotation = signature.return_annotation
    assert return_annotation == Dict[str, Any] or str(return_annotation) == 'typing.Dict[str, typing.Any]'


def test_air_change_dataset_version_dict_functionality(air_change_dataset_train):
    """Test that AirChangeDataset version dict method works correctly."""
    
    version_dict = air_change_dataset_train._get_cache_version_dict()
    
    # Should return a dictionary
    assert isinstance(version_dict, dict)
    
    # Should contain class_name
    assert 'class_name' in version_dict
    assert version_dict['class_name'] == 'AirChangeDataset'
    
    # Should contain base parameters
    assert 'data_root' in version_dict
    assert 'split' in version_dict
    assert version_dict['split'] == 'train'
    
    # Should contain AirChangeDataset specific parameters
    assert 'train_crops_per_image' in version_dict
    assert 'image_size' in version_dict
    assert 'test_crop_size' in version_dict
    assert 'train_crop_size' in version_dict
    
    # Verify values match class constants
    assert version_dict['train_crops_per_image'] == AirChangeDataset.TRAIN_CROPS_PER_IMAGE
    assert version_dict['image_size'] == AirChangeDataset.IMAGE_SIZE
    assert version_dict['test_crop_size'] == AirChangeDataset.TEST_CROP_SIZE
    assert version_dict['train_crop_size'] == AirChangeDataset.TRAIN_CROP_SIZE
