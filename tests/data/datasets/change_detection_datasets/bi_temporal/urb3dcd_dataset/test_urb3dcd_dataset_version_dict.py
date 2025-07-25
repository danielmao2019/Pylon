"""Test version dict implementation for Urb3DCDDataset."""

import pytest
import tempfile
import os
import numpy as np
from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset



def test_urb3dcd_dataset_has_version_dict_method(create_dummy_urb3dcd_files):
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


def test_urb3dcd_dataset_version_dict_functionality(create_dummy_urb3dcd_files):
    """Test that Urb3DCDDataset version dict method works correctly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_urb3dcd_files(temp_dir)
        
        dataset = Urb3DCDDataset(
            data_root=temp_dir, 
            split='train',
            version=1,
            patched=True,
            sample_per_epoch=128,
            fix_samples=False,
            radius=50.0
        )
        version_dict = dataset._get_cache_version_dict()
        
        # Should return a dictionary
        assert isinstance(version_dict, dict)
        
        # Should contain class_name
        assert 'class_name' in version_dict
        assert version_dict['class_name'] == 'Urb3DCDDataset'
        
        # Should contain base parameters
        assert 'data_root' in version_dict
        assert 'split' in version_dict
        assert version_dict['split'] == 'train'
        
        # Should contain Urb3DCDDataset specific parameters
        assert 'version' in version_dict
        assert 'patched' in version_dict
        assert 'sample_per_epoch' in version_dict
        assert 'fix_samples' in version_dict
        assert 'radius' in version_dict
        
        # Verify values match constructor parameters
        assert version_dict['version'] == 1
        assert version_dict['patched'] == True
        assert version_dict['sample_per_epoch'] == 128
        assert version_dict['fix_samples'] == False
        assert version_dict['radius'] == 50.0