"""Test version dict implementation for CelebADataset."""

import pytest
import tempfile
import os
import json
from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset




def test_celeb_a_dataset_has_version_dict_method():
    """Test that CelebADataset has _get_cache_version_dict method."""
    assert hasattr(CelebADataset, '_get_cache_version_dict')
    
    # Check method signature
    import inspect
    method = getattr(CelebADataset, '_get_cache_version_dict')
    signature = inspect.signature(method)
    
    # Should take only self parameter
    params = list(signature.parameters.keys())
    assert params == ['self']
    
    # Should return Dict[str, Any]
    from typing import Dict, Any
    return_annotation = signature.return_annotation
    assert return_annotation == Dict[str, Any] or str(return_annotation) == 'typing.Dict[str, typing.Any]'


def test_celeb_a_dataset_version_dict_functionality(create_dummy_celeb_a_files):
    """Test that CelebADataset version dict method works correctly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_celeb_a_files(temp_dir)
        
        dataset = CelebADataset(data_root=temp_dir, split='train', use_landmarks=True)
        version_dict = dataset._get_cache_version_dict()
        
        # Should return a dictionary
        assert isinstance(version_dict, dict)
        
        # Should contain class_name
        assert 'class_name' in version_dict
        assert version_dict['class_name'] == 'CelebADataset'
        
        # Should contain CelebA-specific parameters
        assert 'use_landmarks' in version_dict
        assert version_dict['use_landmarks'] == True