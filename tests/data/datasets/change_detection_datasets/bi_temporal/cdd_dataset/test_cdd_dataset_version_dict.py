"""Test version dict implementation for CDDDataset."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.cdd_dataset import CDDDataset


def test_cdd_dataset_has_version_dict_method():
    """Test that CDDDataset has _get_cache_version_dict method."""
    assert hasattr(CDDDataset, '_get_cache_version_dict')
    
    # Check method signature
    import inspect
    method = getattr(CDDDataset, '_get_cache_version_dict')
    signature = inspect.signature(method)
    
    # Should take only self parameter
    params = list(signature.parameters.keys())
    assert params == ['self']
    
    # Should return Dict[str, Any]
    from typing import Dict, Any
    return_annotation = signature.return_annotation
    assert return_annotation == Dict[str, Any] or str(return_annotation) == 'typing.Dict[str, typing.Any]'


def test_cdd_dataset_version_dict_functionality(cdd_dataset_train):
    """Test that CDDDataset version dict method works correctly."""
    
    version_dict = cdd_dataset_train._get_cache_version_dict()
    
    # Should return a dictionary
    assert isinstance(version_dict, dict)
    
    # Should contain class_name
    assert 'class_name' in version_dict
    assert version_dict['class_name'] == 'CDDDataset'
    
    # Should contain base parameters (data_root intentionally excluded for cache stability)
    assert 'split' in version_dict
    assert version_dict['split'] == 'train'
