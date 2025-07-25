"""Test version dict implementation for SYSU_CD_Dataset."""

import pytest
import tempfile
import os
from data.datasets.change_detection_datasets.bi_temporal.sysu_cd_dataset import SYSU_CD_Dataset



def test_sysu_cd_dataset_has_version_dict_method(create_dummy_sysu_cd_files):
    """Test that SYSU_CD_Dataset has _get_cache_version_dict method."""
    assert hasattr(SYSU_CD_Dataset, '_get_cache_version_dict')
    
    # Check method signature
    import inspect
    method = getattr(SYSU_CD_Dataset, '_get_cache_version_dict')
    signature = inspect.signature(method)
    
    # Should take only self parameter
    params = list(signature.parameters.keys())
    assert params == ['self']
    
    # Should return Dict[str, Any]
    from typing import Dict, Any
    return_annotation = signature.return_annotation
    assert return_annotation == Dict[str, Any] or str(return_annotation) == 'typing.Dict[str, typing.Any]'


def test_sysu_cd_dataset_version_dict_functionality(create_dummy_sysu_cd_files):
    """Test that SYSU_CD_Dataset version dict method works correctly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_sysu_cd_files(temp_dir)
        
        dataset = SYSU_CD_Dataset(data_root=temp_dir, split='train')
        version_dict = dataset._get_cache_version_dict()
        
        # Should return a dictionary
        assert isinstance(version_dict, dict)
        
        # Should contain class_name
        assert 'class_name' in version_dict
        assert version_dict['class_name'] == 'SYSU_CD_Dataset'
        
        # Should contain base parameters
        assert 'data_root' in version_dict
        assert 'split' in version_dict
        assert version_dict['split'] == 'train'