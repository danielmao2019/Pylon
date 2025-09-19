"""Tests for base display methods in dataset classes.

This module tests that all base display classes implement the display_datapoint
method correctly and return proper HTML structures using real production datasets.
No mocking is used - all tests use actual production dataset classes that can be
instantiated without external data files.
"""
import pytest
import os
import tempfile
import torch
import numpy as np
from typing import Dict, Any
from dash import html
from data.datasets.semantic_segmentation_datasets.base_semseg_dataset import BaseSemsegDataset
from data.datasets.change_detection_datasets.base_2dcd_dataset import Base2DCDDataset
from data.datasets.change_detection_datasets.base_3dcd_dataset import Base3DCDDataset
from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset

# Import real production dataset classes
from data.datasets.semantic_segmentation_datasets.whu_bd_dataset import WHU_BD_Dataset
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset
from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset
from data.datasets.pcr_datasets.modelnet40_dataset import ModelNet40Dataset
from utils.builders import build_from_config


# Real production dataset fixtures using actual dataset classes
# No mocking - all datasets use production implementations

@pytest.fixture(autouse=True)
def setup_registry_for_point_cloud_utils():
    """Set up minimal registry only for point cloud utility functions.
    
    This is ONLY needed because point_cloud.py utilities access registry.viewer.backend.current_dataset
    for generating point cloud IDs. We use a minimal setup but with real infrastructure.
    """
    from data.viewer.callbacks import registry
    
    # Only create if doesn't exist
    if not hasattr(registry, 'viewer') or registry.viewer is None:
        # Create minimal backend that only provides current_dataset attribute
        class MinimalBackend:
            def __init__(self):
                self.current_dataset = 'test_dataset'
        
        # Create minimal viewer
        class MinimalViewer:
            def __init__(self):
                self.backend = MinimalBackend()
        
        # Store original state
        original_viewer = getattr(registry, 'viewer', None)
        
        # Set minimal viewer
        registry.viewer = MinimalViewer()
        
        yield
        
        # Restore original state
        if original_viewer is not None:
            registry.viewer = original_viewer
        elif hasattr(registry, 'viewer'):
            delattr(registry, 'viewer')
    else:
        # Registry already exists, just yield
        yield


# No temp data creation needed - use real datasets with real data paths


# We now use real production datasets directly - no TestableXXX classes needed!


@pytest.fixture
def semseg_dataset(whu_bd_data_root):
    """Real semantic segmentation dataset using WHU_BD_Dataset directly."""
    return WHU_BD_Dataset(data_root=whu_bd_data_root, split='test')
    

@pytest.fixture
def valid_semseg_datapoint(semseg_dataset):
    """Valid semantic segmentation datapoint from real production dataset."""
    # Get actual datapoint from real dataset instance
    datapoint = semseg_dataset[0]
    
    # Validate that it has the expected structure for semantic segmentation
    assert 'inputs' in datapoint, "Datapoint must have 'inputs' key"
    assert 'labels' in datapoint, "Datapoint must have 'labels' key"
    assert 'meta_info' in datapoint, "Datapoint must have 'meta_info' key"
    assert 'image' in datapoint['inputs'], "Inputs must have 'image' key"
    assert 'semantic_map' in datapoint['labels'], "Labels must have 'semantic_map' key"  # WHU uses semantic_map
    
    # Convert semantic_map to label for base class compatibility
    datapoint['labels']['label'] = datapoint['labels']['semantic_map']
    
    return datapoint


# Real OSCD dataset - no modifications needed!


@pytest.fixture
def cd2d_dataset(oscd_data_root):
    """Real 2D change detection dataset using OSCDDataset directly."""
    return OSCDDataset(data_root=oscd_data_root, split='test')


@pytest.fixture
def valid_2dcd_datapoint(cd2d_dataset):
    """Valid 2D change detection datapoint from real production dataset."""
    # Get actual datapoint from real dataset instance
    datapoint = cd2d_dataset[0]
    
    # Validate that it has the expected structure for 2D change detection
    assert 'inputs' in datapoint, "Datapoint must have 'inputs' key"
    assert 'labels' in datapoint, "Datapoint must have 'labels' key"
    assert 'meta_info' in datapoint, "Datapoint must have 'meta_info' key"
    assert 'img_1' in datapoint['inputs'], "Inputs must have 'img_1' key"
    assert 'img_2' in datapoint['inputs'], "Inputs must have 'img_2' key"
    assert 'change_map' in datapoint['labels'], "Labels must have 'change_map' key"
    
    return datapoint




@pytest.fixture
def cd3d_dataset(urb3dcd_data_root):
    """Real 3D change detection dataset using Urb3DCDDataset directly."""
    # Use patched=False to avoid cylinder sampling issues, and explicitly set radius to default
    return Urb3DCDDataset(data_root=urb3dcd_data_root, split='test', patched=False, radius=20)


@pytest.fixture
def valid_3dcd_datapoint(cd3d_dataset):
    """Valid 3D change detection datapoint from real production dataset."""
    # Get actual datapoint from real dataset instance
    datapoint = cd3d_dataset[0]
    
    # Validate that it has the expected structure for 3D change detection
    assert 'inputs' in datapoint, "Datapoint must have 'inputs' key"
    assert 'labels' in datapoint, "Datapoint must have 'labels' key"
    assert 'meta_info' in datapoint, "Datapoint must have 'meta_info' key"
    assert 'pc_1' in datapoint['inputs'], "Inputs must have 'pc_1' key"
    assert 'pc_2' in datapoint['inputs'], "Inputs must have 'pc_2' key"
    assert 'change_map' in datapoint['labels'], "Labels must have 'change_map' key"
    
    return datapoint




@pytest.fixture
def pcr_dataset(modelnet40_data_root):
    """Real PCR dataset using ModelNet40Dataset directly."""
    return ModelNet40Dataset(data_root=modelnet40_data_root, dataset_size=100)


@pytest.fixture
def valid_pcr_datapoint(pcr_dataset):
    """Valid point cloud registration datapoint from real production dataset."""
    # Get actual datapoint from real dataset instance
    datapoint = pcr_dataset[0]
    
    # Validate that it has the expected structure for PCR
    assert 'inputs' in datapoint, "Datapoint must have 'inputs' key"
    assert 'labels' in datapoint, "Datapoint must have 'labels' key"
    assert 'meta_info' in datapoint, "Datapoint must have 'meta_info' key"
    assert 'src_pc' in datapoint['inputs'], "Inputs must have 'src_pc' key"
    assert 'tgt_pc' in datapoint['inputs'], "Inputs must have 'tgt_pc' key"
    assert 'correspondences' in datapoint['inputs'], "Inputs must have 'correspondences' key"
    assert 'transform' in datapoint['labels'], "Labels must have 'transform' key"
    
    return datapoint


@pytest.fixture
def default_camera_state():
    """Default camera state for 3D display tests."""
    return {
        'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5},
        'center': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'up': {'x': 0.0, 'y': 0.0, 'z': 1.0}
    }


@pytest.fixture
def default_3d_settings():
    """Default 3D settings for display tests."""
    return {
        'point_size': 2.0,
        'point_opacity': 0.8,
        'lod_type': 'continuous'
    }


# Base display classes existence tests


def test_base_display_classes_have_display_method():
    """Test that all base display classes implement display_datapoint method."""
    base_classes = [
        BaseSemsegDataset,
        Base2DCDDataset,
        Base3DCDDataset,
        BasePCRDataset
    ]
    
    for base_class in base_classes:
        # Check that display_datapoint method exists
        assert hasattr(base_class, 'display_datapoint'), f"{base_class.__name__} must have display_datapoint method"
        
        # Check that it's a static method (callable on class)
        assert callable(getattr(base_class, 'display_datapoint')), f"{base_class.__name__}.display_datapoint must be callable"
        
        # Check method signature - should accept required parameters
        method = getattr(base_class, 'display_datapoint')
        import inspect
        signature = inspect.signature(method)
        
        # Should have these parameters
        expected_params = ['datapoint', 'class_labels', 'camera_state', 'settings_3d']
        actual_params = list(signature.parameters.keys())
        
        for param in expected_params:
            assert param in actual_params, f"{base_class.__name__}.display_datapoint must have parameter '{param}'"


# Display method functionality tests


def test_semseg_display_method_returns_html_div(valid_semseg_datapoint):
    """Test semantic segmentation display method returns proper HTML."""
    result = BaseSemsegDataset.display_datapoint(valid_semseg_datapoint)
    
    # Should return html.Div instance
    assert isinstance(result, html.Div), f"Expected html.Div, got {type(result)}"
    
    # Should have children (content)
    assert hasattr(result, 'children'), "HTML Div should have children attribute"
    assert result.children is not None, "HTML Div children should not be None"
    assert len(result.children) > 0, "HTML Div should have content"


def test_2dcd_display_method_returns_html_div(valid_2dcd_datapoint):
    """Test 2D change detection display method returns proper HTML."""
    result = Base2DCDDataset.display_datapoint(valid_2dcd_datapoint)
    
    # Should return html.Div instance
    assert isinstance(result, html.Div), f"Expected html.Div, got {type(result)}"
    
    # Should have children (content)
    assert hasattr(result, 'children'), "HTML Div should have children attribute"
    assert result.children is not None, "HTML Div children should not be None"
    assert len(result.children) > 0, "HTML Div should have content"


def test_3dcd_display_method_returns_html_div(valid_3dcd_datapoint, default_camera_state):
    """Test 3D change detection display method returns proper HTML."""
    # Use 'none' LOD to avoid point count mismatches during testing
    test_3d_settings = {
        'point_size': 2.0,
        'point_opacity': 0.8,
        'lod_type': 'none'
    }
    result = Base3DCDDataset.display_datapoint(
        datapoint=valid_3dcd_datapoint,
        camera_state=default_camera_state,
        settings_3d=test_3d_settings
    )
    
    # Should return html.Div instance
    assert isinstance(result, html.Div), f"Expected html.Div, got {type(result)}"
    
    # Should have children (content)
    assert hasattr(result, 'children'), "HTML Div should have children attribute"
    assert result.children is not None, "HTML Div children should not be None"
    assert len(result.children) > 0, "HTML Div should have content"


def test_pcr_display_method_returns_html_div(valid_pcr_datapoint, default_camera_state):
    """Test point cloud registration display method returns proper HTML."""
    # Use 'none' LOD to avoid LOD-related color/point mismatches during testing
    test_3d_settings = {
        'point_size': 2.0,
        'point_opacity': 0.8,
        'lod_type': 'none'
    }
    result = BasePCRDataset.display_datapoint(
        datapoint=valid_pcr_datapoint,
        camera_state=default_camera_state,
        settings_3d=test_3d_settings
    )
    
    # Should return html.Div instance
    assert isinstance(result, html.Div), f"Expected html.Div, got {type(result)}"
    
    # Should have children (content)
    assert hasattr(result, 'children'), "HTML Div should have children attribute"
    assert result.children is not None, "HTML Div children should not be None"
    assert len(result.children) > 0, "HTML Div should have content"


# Display method parameter handling tests


def test_display_methods_with_optional_parameters(semseg_dataset, cd2d_dataset, cd3d_dataset, pcr_dataset):
    """Test display methods handle optional parameters correctly."""
    # Get real datapoints from actual dataset instances
    semseg_datapoint = semseg_dataset[0]
    # Fix semantic_map -> label key conversion for WHU dataset compatibility
    if 'semantic_map' in semseg_datapoint['labels']:
        semseg_datapoint['labels']['label'] = semseg_datapoint['labels']['semantic_map']
    
    cd2d_datapoint = cd2d_dataset[0]
    cd3d_datapoint = cd3d_dataset[0]
    pcr_datapoint = pcr_dataset[0]
    
    test_cases = [
        (BaseSemsegDataset, semseg_datapoint),
        (Base2DCDDataset, cd2d_datapoint),
        (Base3DCDDataset, cd3d_datapoint),
        (BasePCRDataset, pcr_datapoint)
    ]
    
    optional_params = {
        'class_labels': {'0': ['background'], '1': ['class1']},
        'camera_state': {
            'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5},
            'center': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'up': {'x': 0.0, 'y': 0.0, 'z': 1.0}
        },
        'settings_3d': {'point_size': 2.0, 'point_opacity': 0.8, 'lod_type': 'continuous'}
    }
    
    for base_class, datapoint in test_cases:
        # Test with all optional parameters provided
        result = base_class.display_datapoint(
            datapoint=datapoint,
            class_labels=optional_params['class_labels'],
            camera_state=optional_params['camera_state'],
            settings_3d=optional_params['settings_3d']
        )
        
        assert isinstance(result, html.Div), f"{base_class.__name__} should return html.Div with optional parameters"
        
        # Test with no optional parameters (handle 3D special case)
        if base_class in [Base3DCDDataset, BasePCRDataset]:
            # For 3D datasets, use lod_type="none" when camera_state is None
            none_settings_3d = {'lod_type': 'none', 'density_percentage': 99}
            result = base_class.display_datapoint(
                datapoint=datapoint,
                class_labels=None,
                camera_state=None,
                settings_3d=none_settings_3d
            )
        else:
            # For 2D datasets, all None is fine
            result = base_class.display_datapoint(
                datapoint=datapoint,
                class_labels=None,
                camera_state=None,
                settings_3d=None
            )
        
        assert isinstance(result, html.Div), f"{base_class.__name__} should return html.Div with None parameters"


# Error handling tests


def test_display_methods_with_invalid_input():
    """Test display methods handle invalid input appropriately."""
    
    # Test with None datapoint
    with pytest.raises(AssertionError) as exc_info:
        BaseSemsegDataset.display_datapoint(None)
    
    assert "datapoint must not be None" in str(exc_info.value)
    
    # Test with non-dict datapoint
    with pytest.raises(AssertionError) as exc_info:
        BaseSemsegDataset.display_datapoint("not_a_dict")
    
    assert "datapoint must be dict" in str(exc_info.value)
    
    # Test with invalid structure - should raise validation error
    invalid_datapoint = {
        'inputs': {},  # Empty inputs should fail validation
        'labels': {'semantic_map': torch.randint(0, 10, (32, 32))},  # Use semantic_map for WHU
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError):
        BaseSemsegDataset.display_datapoint(invalid_datapoint)


def test_display_methods_validation_integration(semseg_dataset, cd2d_dataset, cd3d_dataset, pcr_dataset):
    """Test that display methods properly integrate with structure validation."""
    
    # Test each display method with its corresponding invalid structure
    # Create invalid structures by modifying real datapoints
    
    # Semantic segmentation with missing image
    invalid_semseg = semseg_dataset[0]
    # Fix semantic_map -> label key conversion for WHU dataset compatibility
    if 'semantic_map' in invalid_semseg['labels']:
        invalid_semseg['labels']['label'] = invalid_semseg['labels']['semantic_map']
    invalid_semseg = {
        'inputs': {'not_image': invalid_semseg['inputs']['image']},
        'labels': invalid_semseg['labels'],
        'meta_info': invalid_semseg['meta_info']
    }
    
    # 2D change detection with missing img_2
    invalid_2dcd = cd2d_dataset[0]
    invalid_2dcd = {
        'inputs': {'img_1': invalid_2dcd['inputs']['img_1']},
        'labels': invalid_2dcd['labels'],
        'meta_info': invalid_2dcd['meta_info']
    }
    
    # 3D change detection with missing pc_2
    invalid_3dcd = cd3d_dataset[0]
    invalid_3dcd = {
        'inputs': {'pc_1': invalid_3dcd['inputs']['pc_1']},
        'labels': invalid_3dcd['labels'],
        'meta_info': invalid_3dcd['meta_info']
    }
    
    # PCR with missing tgt_pc
    invalid_pcr = pcr_dataset[0]
    invalid_pcr = {
        'inputs': {'src_pc': invalid_pcr['inputs']['src_pc']},
        'labels': invalid_pcr['labels'],
        'meta_info': invalid_pcr['meta_info']
    }
    
    invalid_cases = [
        (BaseSemsegDataset, invalid_semseg),
        (Base2DCDDataset, invalid_2dcd),
        (Base3DCDDataset, invalid_3dcd),
        (BasePCRDataset, invalid_pcr)
    ]
    
    for base_class, invalid_datapoint in invalid_cases:
        with pytest.raises(AssertionError):
            base_class.display_datapoint(invalid_datapoint)


# Content structure tests


def test_display_methods_generate_meaningful_content(semseg_dataset, cd2d_dataset):
    """Test that display methods generate meaningful content structures."""
    
    # Get real datapoints from actual dataset instances
    semseg_datapoint = semseg_dataset[0]
    # Fix semantic_map -> label key conversion for WHU dataset compatibility
    if 'semantic_map' in semseg_datapoint['labels']:
        semseg_datapoint['labels']['label'] = semseg_datapoint['labels']['semantic_map']
    
    cd2d_datapoint = cd2d_dataset[0]
    
    test_cases = [
        (BaseSemsegDataset, semseg_datapoint),
        (Base2DCDDataset, cd2d_datapoint)
    ]
    
    for base_class, datapoint in test_cases:
        result = base_class.display_datapoint(datapoint)
        
        # Should be an html.Div with meaningful structure
        assert isinstance(result, html.Div)
        assert result.children is not None
        assert len(result.children) > 0
        
        # Convert to string to check for basic content presence
        # This is a basic smoke test to ensure the display methods are working
        try:
            html_str = str(result)
            assert len(html_str) > 100, f"Display output seems too short for {base_class.__name__}: {len(html_str)} chars"
        except Exception as e:
            pytest.fail(f"Failed to convert {base_class.__name__} display output to string: {e}")


def test_display_methods_handle_different_tensor_dtypes(semseg_dataset):
    """Test display methods handle different tensor dtypes correctly."""
    
    # Get real datapoint from actual dataset instance
    datapoint = semseg_dataset[0]
    # Fix semantic_map -> label key conversion for WHU dataset compatibility
    if 'semantic_map' in datapoint['labels']:
        datapoint['labels']['label'] = datapoint['labels']['semantic_map']
    
    # Test with the dtype from real dataset (float32)
    result = BaseSemsegDataset.display_datapoint(datapoint)
    assert isinstance(result, html.Div), "Failed with real dataset dtype"
    assert result.children is not None, "Empty result with real dataset dtype"
