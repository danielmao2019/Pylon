"""Test version dict implementation for KC3DDataset."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.kc_3d_dataset import KC3DDataset


def test_kc3d_dataset_has_version_dict_method():
    """Test that KC3DDataset has _get_cache_version_dict method."""
    assert hasattr(KC3DDataset, '_get_cache_version_dict')
    
    # Check method signature
    import inspect
    method = getattr(KC3DDataset, '_get_cache_version_dict')
    signature = inspect.signature(method)
    
    # Should take only self parameter
    params = list(signature.parameters.keys())
    assert params == ['self']
    
    # Should return Dict[str, Any]
    from typing import Dict, Any
    return_annotation = signature.return_annotation
    assert return_annotation == Dict[str, Any] or str(return_annotation) == 'typing.Dict[str, typing.Any]'


def test_kc3d_dataset_version_dict_functionality(kc_3d_dataset_train):
    """Test that KC3DDataset version dict method works correctly."""
    
    version_dict = kc_3d_dataset_train._get_cache_version_dict()
    
    # Should return a dictionary
    assert isinstance(version_dict, dict)
    
    # Should contain class_name
    assert 'class_name' in version_dict
    assert version_dict['class_name'] == 'KC3DDataset'
    
    # Should contain base parameters
    # NOTE: data_root is intentionally excluded for cache stability across different paths
    assert 'split' in version_dict
    assert version_dict['split'] == 'train'
    
    # Should contain KC3DDataset specific parameters
    assert 'use_ground_truth_registration' in version_dict
    
    # Verify parameter has the expected default value 
    # (The original test expected True as the default when not specified)
    assert version_dict['use_ground_truth_registration'] == True


def test_kc3d_dataset_version_dict_parameter_variations(kc_3d_data_root):
    """Test that different use_ground_truth_registration values produce different version dicts."""
    
    # Test with use_ground_truth_registration=True
    dataset_with_gt = KC3DDataset(data_root=kc_3d_data_root, split='train', use_ground_truth_registration=True)
    version_dict_with_gt = dataset_with_gt._get_cache_version_dict()
    assert version_dict_with_gt['use_ground_truth_registration'] == True
    
    # Test with use_ground_truth_registration=False  
    dataset_no_gt = KC3DDataset(data_root=kc_3d_data_root, split='train', use_ground_truth_registration=False)
    version_dict_no_gt = dataset_no_gt._get_cache_version_dict()
    assert version_dict_no_gt['use_ground_truth_registration'] == False
