"""Test version dict implementation for I3PEDataset."""

import pytest
import tempfile
import os
from data.datasets.change_detection_datasets.single_temporal.i3pe_dataset import I3PEDataset
from data.datasets.random_datasets.classification_random_dataset import ClassificationRandomDataset


def test_i3pe_dataset_has_version_dict_method():
    """Test that I3PEDataset has _get_cache_version_dict method."""
    assert hasattr(I3PEDataset, '_get_cache_version_dict')
    
    # Check method signature
    import inspect
    method = getattr(I3PEDataset, '_get_cache_version_dict')
    signature = inspect.signature(method)
    
    # Should take only self parameter
    params = list(signature.parameters.keys())
    assert params == ['self']
    
    # Should return Dict[str, Any]
    from typing import Dict, Any
    return_annotation = signature.return_annotation
    assert return_annotation == Dict[str, Any] or str(return_annotation) == 'typing.Dict[str, typing.Any]'


def test_i3pe_dataset_version_dict_functionality():
    """Test that I3PEDataset version dict method works correctly."""
    
    # Create a simple source dataset
    source = ClassificationRandomDataset(
        num_examples=10,
        num_classes=5,
        image_res=(64, 64),
        initial_seed=42
    )
    
    dataset = I3PEDataset(
        source=source,
        exchange_ratio=0.75
    )
    version_dict = dataset._get_cache_version_dict()
    
    # Should return a dictionary
    assert isinstance(version_dict, dict)
    
    # Should contain class_name
    assert 'class_name' in version_dict
    assert version_dict['class_name'] == 'I3PEDataset'
    
    # Should contain base synthetic dataset parameters
    assert 'source_class_name' in version_dict
    assert version_dict['source_class_name'] == 'ClassificationRandomDataset'
    
    # Should contain I3PEDataset specific parameters
    assert 'exchange_ratio' in version_dict
    assert 'n_segments' in version_dict
    assert 'eps' in version_dict
    assert 'min_samples' in version_dict
    assert 'scale_factors' in version_dict
    
    # Verify values match constructor and class attributes
    assert version_dict['exchange_ratio'] == 0.75
    assert version_dict['n_segments'] == I3PEDataset.n_segments
    assert version_dict['eps'] == I3PEDataset.eps
    assert version_dict['min_samples'] == I3PEDataset.min_samples
    assert version_dict['scale_factors'] == I3PEDataset.scale_factors