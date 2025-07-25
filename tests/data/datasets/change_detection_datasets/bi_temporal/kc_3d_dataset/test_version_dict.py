"""Test version dict implementation for KC3DDataset."""

import pytest
import tempfile
import os
import pickle
import numpy as np
from data.datasets.change_detection_datasets.bi_temporal.kc_3d_dataset import KC3DDataset



def test_kc3d_dataset_has_version_dict_method(create_dummy_kc3d_files):
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


def test_kc3d_dataset_version_dict_functionality(create_dummy_kc3d_files):
    """Test that KC3DDataset version dict method works correctly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_kc3d_files(temp_dir)
        
        dataset = KC3DDataset(data_root=temp_dir, split='train', use_ground_truth_registration=True)
        version_dict = dataset._get_cache_version_dict()
        
        # Should return a dictionary
        assert isinstance(version_dict, dict)
        
        # Should contain class_name
        assert 'class_name' in version_dict
        assert version_dict['class_name'] == 'KC3DDataset'
        
        # Should contain base parameters
        assert 'data_root' in version_dict
        assert 'split' in version_dict
        assert version_dict['split'] == 'train'
        
        # Should contain KC3DDataset specific parameters
        assert 'use_ground_truth_registration' in version_dict
        assert version_dict['use_ground_truth_registration'] == True
        
        # Test with different use_ground_truth_registration value
        dataset_no_gt = KC3DDataset(data_root=temp_dir, split='train', use_ground_truth_registration=False)
        version_dict_no_gt = dataset_no_gt._get_cache_version_dict()
        assert version_dict_no_gt['use_ground_truth_registration'] == False