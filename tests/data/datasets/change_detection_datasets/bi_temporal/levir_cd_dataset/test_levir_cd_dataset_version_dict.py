"""Test version dict implementation for LevirCdDataset."""

import pytest
import os
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset




def test_levir_cd_dataset_has_version_dict_method():
    """Test that LevirCdDataset has _get_cache_version_dict method."""
    assert hasattr(LevirCdDataset, '_get_cache_version_dict')
    
    # Check method signature
    import inspect
    method = getattr(LevirCdDataset, '_get_cache_version_dict')
    signature = inspect.signature(method)
    
    # Should take only self parameter
    params = list(signature.parameters.keys())
    assert params == ['self']
    
    # Should return Dict[str, Any]
    from typing import Dict, Any
    return_annotation = signature.return_annotation
    assert return_annotation == Dict[str, Any] or str(return_annotation) == 'typing.Dict[str, typing.Any]'


def test_levir_cd_dataset_version_dict_functionality(levir_cd_dataset_train):
    """Test that LevirCdDataset version dict method works correctly."""
    
    version_dict = levir_cd_dataset_train._get_cache_version_dict()
    
    # Should return a dictionary
    assert isinstance(version_dict, dict)
    
    # Should contain class_name
    assert 'class_name' in version_dict
    assert version_dict['class_name'] == 'LevirCdDataset'
    
    # Should contain base parameters (data_root intentionally excluded for cache stability)
    assert 'split' in version_dict
    assert version_dict['split'] == 'train'
