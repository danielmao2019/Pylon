"""Test version dict implementation for PPSLDataset."""

import pytest
from data.datasets.change_detection_datasets.single_temporal.ppsl_dataset import PPSLDataset
from data.datasets.random_datasets.semantic_segmentation_random_dataset import SemanticSegmentationRandomDataset


def test_ppsl_dataset_has_version_dict_method():
    """Test that PPSLDataset has _get_cache_version_dict method."""
    assert hasattr(PPSLDataset, '_get_cache_version_dict')
    
    # Check method signature
    import inspect
    method = getattr(PPSLDataset, '_get_cache_version_dict')
    signature = inspect.signature(method)
    
    # Should take only self parameter
    params = list(signature.parameters.keys())
    assert params == ['self']
    
    # Should return Dict[str, Any]
    from typing import Dict, Any
    return_annotation = signature.return_annotation
    assert return_annotation == Dict[str, Any] or str(return_annotation) == 'typing.Dict[str, typing.Any]'


def test_ppsl_dataset_version_dict_functionality():
    """Test that PPSLDataset version dict method works correctly."""
    
    # Create a semantic segmentation source dataset
    source = SemanticSegmentationRandomDataset(
        num_examples=10,
        num_classes=5,
        base_seed=42
    )
    
    dataset = PPSLDataset(source=source, dataset_size=10)
    version_dict = dataset._get_cache_version_dict()
    
    # Should return a dictionary
    assert isinstance(version_dict, dict)
    
    # Should contain class_name
    assert 'class_name' in version_dict
    assert version_dict['class_name'] == 'PPSLDataset'
    
    # Should contain base synthetic dataset parameters
    assert 'source_class' in version_dict
    assert version_dict['source_class'] == 'SemanticSegmentationRandomDataset'
    
    # Should contain PPSLDataset specific transform parameters
    assert 'colorjit_brightness' in version_dict
    assert 'colorjit_contrast' in version_dict
    assert 'colorjit_saturation' in version_dict
    assert 'colorjit_hue' in version_dict
    assert 'affine_degrees' in version_dict
    assert 'affine_scale' in version_dict
    assert 'affine_translate' in version_dict
    assert 'affine_shear' in version_dict
    
    # Verify values match the hardcoded parameters
    assert version_dict['colorjit_brightness'] == 0.7
    assert version_dict['colorjit_contrast'] == 0.7
    assert version_dict['colorjit_saturation'] == 0.7
    assert version_dict['colorjit_hue'] == 0.2
    assert version_dict['affine_degrees'] == (-5, 5)
    assert version_dict['affine_scale'] == (1, 1.02)
    assert version_dict['affine_translate'] == (0.02, 0.02)
    assert version_dict['affine_shear'] == (-5, 5)
