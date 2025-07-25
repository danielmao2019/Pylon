"""Test version dict implementation for Bi2SingleTemporal."""

import pytest
import tempfile
import os
from data.datasets.change_detection_datasets.single_temporal.bi2single_temporal_dataset import Bi2SingleTemporal
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset



def test_bi2single_temporal_has_version_dict_method(create_dummy_levir_cd_files):
    """Test that Bi2SingleTemporal has _get_cache_version_dict method."""
    assert hasattr(Bi2SingleTemporal, '_get_cache_version_dict')
    
    # Check method signature
    import inspect
    method = getattr(Bi2SingleTemporal, '_get_cache_version_dict')
    signature = inspect.signature(method)
    
    # Should take only self parameter
    params = list(signature.parameters.keys())
    assert params == ['self']
    
    # Should return Dict[str, Any]
    from typing import Dict, Any
    return_annotation = signature.return_annotation
    assert return_annotation == Dict[str, Any] or str(return_annotation) == 'typing.Dict[str, typing.Any]'


def test_bi2single_temporal_version_dict_functionality(create_dummy_levir_cd_files):
    """Test that Bi2SingleTemporal version dict method works correctly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_levir_cd_files(temp_dir)
        
        # Create source dataset
        source = LevirCdDataset(data_root=temp_dir, split='train')
        
        # Create Bi2SingleTemporal dataset
        dataset = Bi2SingleTemporal(source=source)
        version_dict = dataset._get_cache_version_dict()
        
        # Should return a dictionary
        assert isinstance(version_dict, dict)
        
        # Should contain class_name
        assert 'class_name' in version_dict
        assert version_dict['class_name'] == 'Bi2SingleTemporal'
        
        # Should contain base synthetic dataset parameters
        assert 'source_class_name' in version_dict
        assert version_dict['source_class_name'] == 'LevirCdDataset'
        
        # Should contain Bi2SingleTemporal specific parameters
        assert 'source_version' in version_dict
        assert isinstance(version_dict['source_version'], str)
        assert len(version_dict['source_version']) == 16  # xxhash format