"""Test version dict implementation for Urb3DCDDataset."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset
from utils.builders.builder import build_from_config


def test_urb3dcd_dataset_has_version_dict_method():
    """Test that Urb3DCDDataset has _get_cache_version_dict method."""
    assert hasattr(Urb3DCDDataset, '_get_cache_version_dict')
    
    # Check method signature
    import inspect
    method = getattr(Urb3DCDDataset, '_get_cache_version_dict')
    signature = inspect.signature(method)
    
    # Should take only self parameter
    params = list(signature.parameters.keys())
    assert params == ['self']
    
    # Should return Dict[str, Any]
    from typing import Dict, Any
    return_annotation = signature.return_annotation
    assert return_annotation == Dict[str, Any] or str(return_annotation) == 'typing.Dict[str, typing.Any]'


def test_urb3dcd_dataset_version_dict_with_train_config(urb3dcd_dataset_train_config):
    """Test that Urb3DCDDataset version dict method works correctly."""
    urb3dcd_dataset_train = build_from_config(urb3dcd_dataset_train_config)
    
    version_dict = urb3dcd_dataset_train._get_cache_version_dict()
    
    # Should return a dictionary
    assert isinstance(version_dict, dict)
    
    # Should contain class_name
    assert 'class_name' in version_dict
    assert version_dict['class_name'] == 'Urb3DCDDataset'
    
    # Should contain base parameters (data_root excluded for cache stability)
    assert 'split' in version_dict
    assert version_dict['split'] == 'train'
    
    # data_root should NOT be in version_dict for cache stability across different filesystem locations
    assert 'data_root' not in version_dict
    
    # Should contain Urb3DCDDataset specific parameters
    assert 'version' in version_dict
    assert 'patched' in version_dict
    assert 'sample_per_epoch' in version_dict
    assert 'fix_samples' in version_dict
    assert 'radius' in version_dict
    
    # Verify values match fixture parameters
    assert version_dict['sample_per_epoch'] == 100
    assert version_dict['fix_samples'] == False
    assert version_dict['radius'] == 100
    
    # Verify default values for unspecified parameters
    # Note: These use dataset defaults since not specified in fixture
    assert version_dict['version'] == 1
    assert version_dict['patched'] == True
