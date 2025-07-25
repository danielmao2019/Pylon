"""Test version dict implementation for LevirCdDataset."""

import pytest
import tempfile
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


def test_levir_cd_dataset_version_dict_functionality(create_dummy_levir_structure):
    """Test that LevirCdDataset version dict method works correctly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_levir_structure(temp_dir)
        
        dataset = LevirCdDataset(data_root=temp_dir, split='train')
        version_dict = dataset._get_cache_version_dict()
        
        # Should return a dictionary
        assert isinstance(version_dict, dict)
        
        # Should contain class_name
        assert 'class_name' in version_dict
        assert version_dict['class_name'] == 'LevirCdDataset'
        
        # Should contain base parameters
        assert 'data_root' in version_dict
        assert 'split' in version_dict
        assert version_dict['split'] == 'train'