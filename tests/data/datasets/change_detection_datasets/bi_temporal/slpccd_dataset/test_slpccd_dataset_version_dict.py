"""Test version dict implementation for SLPCCDDataset."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset


def test_slpccd_dataset_has_version_dict_method():
    """Test that SLPCCDDataset has _get_cache_version_dict method."""
    assert hasattr(SLPCCDDataset, '_get_cache_version_dict')
    
    # Check method signature
    import inspect
    method = getattr(SLPCCDDataset, '_get_cache_version_dict')
    signature = inspect.signature(method)
    
    # Should take only self parameter
    params = list(signature.parameters.keys())
    assert params == ['self']
    
    # Should return Dict[str, Any]
    from typing import Dict, Any
    return_annotation = signature.return_annotation
    assert return_annotation == Dict[str, Any] or str(return_annotation) == 'typing.Dict[str, typing.Any]'


def test_slpccd_dataset_version_dict_functionality(slpccd_dataset_train):
    """Test that SLPCCDDataset version dict method works correctly."""
    
    version_dict = slpccd_dataset_train._get_cache_version_dict()
    
    # Should return a dictionary
    assert isinstance(version_dict, dict)
    
    # Should contain class_name
    assert 'class_name' in version_dict
    assert version_dict['class_name'] == 'SLPCCDDataset'
    
    # Should contain base parameters (data_root intentionally excluded for cache stability)
    assert 'split' in version_dict
    assert version_dict['split'] == 'train'
    
    # Should contain SLPCCDDataset specific parameters
    assert 'num_points' in version_dict
    assert 'random_subsample' in version_dict
    assert 'use_hierarchy' in version_dict
    assert 'hierarchy_levels' in version_dict
    assert 'knn_size' in version_dict
    assert 'cross_knn_size' in version_dict
    
    # Verify values match constructor parameters
    assert version_dict['num_points'] == 8192
    assert version_dict['random_subsample'] == True
    assert version_dict['use_hierarchy'] == True
    assert version_dict['hierarchy_levels'] == 3
    assert version_dict['knn_size'] == 16
    assert version_dict['cross_knn_size'] == 16
