"""Tests for base display methods in dataset classes.

This module tests that all base display classes implement the display_datapoint
method correctly and return proper HTML structures.
"""
import pytest
import torch
from typing import Dict, Any, Tuple, Optional, List
from dash import html
from data.datasets.semantic_segmentation_datasets.base_semseg_dataset import BaseSemsegDataset
from data.datasets.change_detection_datasets.base_2d_cd_dataset import Base2DCDDataset
from data.datasets.change_detection_datasets.base_3d_cd_dataset import Base3DCDDataset
from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset


# Test fixtures for valid datapoints


@pytest.fixture
def mock_viewer_registry():
    """Mock the viewer registry for 3D display tests that depend on global state."""
    from data.viewer.callbacks import registry
    from data.viewer.backend.backend import ViewerBackend
    
    # Create a mock viewer object with backend
    class MockViewer:
        def __init__(self):
            self.backend = ViewerBackend()
            self.backend.current_dataset = 'test/dataset'
    
    # Store original registry state
    original_viewer = getattr(registry, 'viewer', None)
    
    # Set mock viewer
    registry.viewer = MockViewer()
    
    yield registry.viewer
    
    # Restore original state
    if original_viewer is not None:
        registry.viewer = original_viewer
    else:
        delattr(registry, 'viewer')


@pytest.fixture
def valid_semseg_datapoint():
    """Valid semantic segmentation datapoint for display testing."""
    return {
        'inputs': {
            'image': torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        },
        'labels': {
            'label': torch.randint(0, 10, (64, 64), dtype=torch.long)
        },
        'meta_info': {
            'image_path': '/path/to/image.jpg',
            'index': 0
        }
    }


@pytest.fixture
def valid_2dcd_datapoint():
    """Valid 2D change detection datapoint for display testing."""
    return {
        'inputs': {
            'img_1': torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8),
            'img_2': torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        },
        'labels': {
            'change_map': torch.randint(0, 2, (64, 64), dtype=torch.long)
        },
        'meta_info': {
            'img1_path': '/path/to/img1.jpg',
            'img2_path': '/path/to/img2.jpg',
            'index': 0
        }
    }


@pytest.fixture
def valid_3dcd_datapoint():
    """Valid 3D change detection datapoint for display testing."""
    n_points = 1000
    return {
        'inputs': {
            'pc_1': {
                'pos': torch.randn(n_points, 3, dtype=torch.float32),
                'features': torch.randn(n_points, 64, dtype=torch.float32)
            },
            'pc_2': {
                'pos': torch.randn(n_points, 3, dtype=torch.float32),
                'features': torch.randn(n_points, 64, dtype=torch.float32)
            }
        },
        'labels': {
            'change_map': torch.randint(0, 2, (n_points,), dtype=torch.long)
        },
        'meta_info': {
            'pc1_path': '/path/to/pc1.pcd',
            'pc2_path': '/path/to/pc2.pcd',
            'index': 0
        }
    }


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


@pytest.fixture
def valid_pcr_datapoint():
    """Valid point cloud registration datapoint for display testing."""
    n_points = 1000
    return {
        'inputs': {
            'src_pc': {
                'pos': torch.randn(n_points, 3, dtype=torch.float32),
                'features': torch.randn(n_points, 64, dtype=torch.float32)
            },
            'tgt_pc': {
                'pos': torch.randn(n_points, 3, dtype=torch.float32),
                'features': torch.randn(n_points, 64, dtype=torch.float32)
            },
            'correspondences': torch.randint(0, n_points, (500, 2), dtype=torch.long)
        },
        'labels': {
            'transform': torch.eye(4, dtype=torch.float32)
        },
        'meta_info': {
            'src_path': '/path/to/src.pcd',
            'tgt_path': '/path/to/tgt.pcd',
            'index': 0
        }
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


def test_3dcd_display_method_returns_html_div(valid_3dcd_datapoint, mock_viewer_registry, default_camera_state, default_3d_settings):
    """Test 3D change detection display method returns proper HTML."""
    result = Base3DCDDataset.display_datapoint(
        datapoint=valid_3dcd_datapoint,
        camera_state=default_camera_state,
        settings_3d=default_3d_settings
    )
    
    # Should return html.Div instance
    assert isinstance(result, html.Div), f"Expected html.Div, got {type(result)}"
    
    # Should have children (content)
    assert hasattr(result, 'children'), "HTML Div should have children attribute"
    assert result.children is not None, "HTML Div children should not be None"
    assert len(result.children) > 0, "HTML Div should have content"


def test_pcr_display_method_returns_html_div(valid_pcr_datapoint, mock_viewer_registry, default_camera_state, default_3d_settings):
    """Test point cloud registration display method returns proper HTML."""
    result = BasePCRDataset.display_datapoint(
        datapoint=valid_pcr_datapoint,
        camera_state=default_camera_state,
        settings_3d=default_3d_settings
    )
    
    # Should return html.Div instance
    assert isinstance(result, html.Div), f"Expected html.Div, got {type(result)}"
    
    # Should have children (content)
    assert hasattr(result, 'children'), "HTML Div should have children attribute"
    assert result.children is not None, "HTML Div children should not be None"
    assert len(result.children) > 0, "HTML Div should have content"


# Display method parameter handling tests


def test_display_methods_with_optional_parameters(mock_viewer_registry):
    """Test display methods handle optional parameters correctly."""
    # Create minimal valid datapoints for each type
    semseg_datapoint = {
        'inputs': {'image': torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)},
        'labels': {'label': torch.randint(0, 10, (32, 32), dtype=torch.long)},
        'meta_info': {}
    }
    
    cd2d_datapoint = {
        'inputs': {
            'img_1': torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8),
            'img_2': torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        },
        'labels': {'change_map': torch.randint(0, 2, (32, 32), dtype=torch.long)},
        'meta_info': {}
    }
    
    cd3d_datapoint = {
        'inputs': {
            'pc_1': {'pos': torch.randn(100, 3, dtype=torch.float32)},
            'pc_2': {'pos': torch.randn(100, 3, dtype=torch.float32)}
        },
        'labels': {'change_map': torch.randint(0, 2, (100,), dtype=torch.long)},
        'meta_info': {}
    }
    
    pcr_datapoint = {
        'inputs': {
            'src_pc': {'pos': torch.randn(100, 3, dtype=torch.float32)},
            'tgt_pc': {'pos': torch.randn(100, 3, dtype=torch.float32)}
        },
        'labels': {'transform': torch.eye(4, dtype=torch.float32)},
        'meta_info': {}
    }
    
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
        'labels': {'label': torch.randint(0, 10, (32, 32))},
        'meta_info': {}
    }
    
    with pytest.raises(AssertionError):
        BaseSemsegDataset.display_datapoint(invalid_datapoint)


def test_display_methods_validation_integration():
    """Test that display methods properly integrate with structure validation."""
    
    # Test each display method with its corresponding invalid structure
    invalid_cases = [
        # Semantic segmentation with missing image
        (BaseSemsegDataset, {
            'inputs': {'not_image': torch.randn(3, 32, 32)},
            'labels': {'label': torch.randint(0, 10, (32, 32))},
            'meta_info': {}
        }),
        
        # 2D change detection with missing img_2
        (Base2DCDDataset, {
            'inputs': {'img_1': torch.randn(3, 32, 32)},
            'labels': {'change_map': torch.randint(0, 2, (32, 32))},
            'meta_info': {}
        }),
        
        # 3D change detection with missing pc_2
        (Base3DCDDataset, {
            'inputs': {'pc_1': {'pos': torch.randn(100, 3)}},
            'labels': {'change_map': torch.randint(0, 2, (100,))},
            'meta_info': {}
        }),
        
        # PCR with missing tgt_pc
        (BasePCRDataset, {
            'inputs': {'src_pc': {'pos': torch.randn(100, 3)}},
            'labels': {'transform': torch.eye(4)},
            'meta_info': {}
        })
    ]
    
    for base_class, invalid_datapoint in invalid_cases:
        with pytest.raises(AssertionError):
            base_class.display_datapoint(invalid_datapoint)


# Content structure tests


def test_display_methods_generate_meaningful_content():
    """Test that display methods generate meaningful content structures."""
    
    # Create test datapoints
    semseg_datapoint = {
        'inputs': {'image': torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)},
        'labels': {'label': torch.randint(0, 10, (64, 64), dtype=torch.long)},
        'meta_info': {'test': 'value'}
    }
    
    cd2d_datapoint = {
        'inputs': {
            'img_1': torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8),
            'img_2': torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        },
        'labels': {'change_map': torch.randint(0, 2, (64, 64), dtype=torch.long)},
        'meta_info': {'test': 'value'}
    }
    
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


def test_display_methods_handle_different_tensor_dtypes():
    """Test display methods handle different tensor dtypes correctly."""
    
    # Test with different image dtypes for semantic segmentation
    dtypes_to_test = [torch.uint8, torch.float32]
    
    for dtype in dtypes_to_test:
        # Create appropriate tensor values for the dtype
        if dtype == torch.uint8:
            image_tensor = torch.randint(0, 255, (3, 32, 32), dtype=dtype)
        else:  # float32
            image_tensor = torch.rand(3, 32, 32, dtype=dtype)
        
        datapoint = {
            'inputs': {'image': image_tensor},
            'labels': {'label': torch.randint(0, 10, (32, 32), dtype=torch.long)},
            'meta_info': {}
        }
        
        # Should not raise exception and should return valid HTML
        result = BaseSemsegDataset.display_datapoint(datapoint)
        assert isinstance(result, html.Div), f"Failed with dtype {dtype}"
        assert result.children is not None, f"Empty result with dtype {dtype}"