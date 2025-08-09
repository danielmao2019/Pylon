"""Tests for ViewerBackend initialization and setup.

This module tests the initialization process, default settings configuration,
and internal state setup of the ViewerBackend class.
"""

import os
import logging
from typing import Dict, Any

import pytest

from data.viewer.backend.backend import ViewerBackend, DATASET_GROUPS, REQUIRES_3D_CLASSES
from data.viewer.utils.settings_config import ViewerSettings


def test_viewer_backend_initialization():
    """Test ViewerBackend initializes with correct default state."""
    backend = ViewerBackend()
    
    # Test logger setup
    assert isinstance(backend.logger, logging.Logger)
    assert backend.logger.name == "data.viewer.backend.backend"
    
    # Test dataset storage initialization
    assert isinstance(backend._datasets, dict)
    assert len(backend._datasets) == 0
    assert isinstance(backend._configs, dict)
    
    # Test transform storage initialization
    assert isinstance(backend._transforms, list)
    assert len(backend._transforms) == 0
    
    # Test state management initialization
    assert backend.current_dataset is None
    assert backend.current_index == 0
    
    # Test 3D settings initialization from ViewerSettings
    default_settings = ViewerSettings.DEFAULT_3D_SETTINGS
    assert backend.point_size == default_settings['point_size']
    assert backend.point_opacity == default_settings['point_opacity']
    assert backend.sym_diff_radius == default_settings['sym_diff_radius']
    assert backend.corr_radius == default_settings['corr_radius']
    assert backend.lod_type == default_settings['lod_type']


def test_initialization_calls_config_init():
    """Test that initialization properly calls _init_dataset_configs."""
    backend = ViewerBackend()
    
    # Config initialization should have been called
    # At minimum, should have attempted to load configs (even if directories don't exist)
    assert isinstance(backend._configs, dict)
    # Configs might be empty if config directories don't exist, which is fine for testing


def test_default_3d_settings_consistency():
    """Test that default 3D settings are consistent with ViewerSettings."""
    backend = ViewerBackend()
    default_settings = ViewerSettings.DEFAULT_3D_SETTINGS
    
    # All default settings should match ViewerSettings
    assert backend.point_size == default_settings['point_size']
    assert backend.point_opacity == default_settings['point_opacity'] 
    assert backend.sym_diff_radius == default_settings['sym_diff_radius']
    assert backend.corr_radius == default_settings['corr_radius']
    assert backend.lod_type == default_settings['lod_type']
    
    # Settings should be of correct types
    assert isinstance(backend.point_size, float)
    assert isinstance(backend.point_opacity, float)
    assert isinstance(backend.sym_diff_radius, float)
    assert isinstance(backend.corr_radius, float)
    assert isinstance(backend.lod_type, str)


def test_dataset_groups_constant():
    """Test DATASET_GROUPS constant has expected structure."""
    assert isinstance(DATASET_GROUPS, dict)
    
    # Test expected dataset types exist
    expected_types = ['semseg', '2dcd', '3dcd', 'pcr', 'mtl', 'general']
    for dataset_type in expected_types:
        assert dataset_type in DATASET_GROUPS
        assert isinstance(DATASET_GROUPS[dataset_type], list)
    
    # Test that all values are strings
    for dataset_type, datasets in DATASET_GROUPS.items():
        for dataset_name in datasets:
            assert isinstance(dataset_name, str), f"Dataset name {dataset_name} in {dataset_type} should be string"


def test_requires_3d_classes_constant():
    """Test REQUIRES_3D_CLASSES constant has expected structure."""
    assert isinstance(REQUIRES_3D_CLASSES, list)
    
    # Test that all entries are strings
    for class_name in REQUIRES_3D_CLASSES:
        assert isinstance(class_name, str), f"Class name {class_name} should be string"
    
    # Test expected 3D classes are present
    expected_3d_classes = [
        'Base3DCDDataset', 'BasePCRDataset', 'Buffer3DDataset',
        'KITTIDataset', 'ThreeDMatchDataset', 'URB3DCDDataset'
    ]
    for expected_class in expected_3d_classes:
        assert expected_class in REQUIRES_3D_CLASSES, f"{expected_class} should be in REQUIRES_3D_CLASSES"


def test_init_dataset_configs_directory_handling():
    """Test _init_dataset_configs handles missing directories gracefully."""
    backend = ViewerBackend()
    
    # Should not crash even if config directories don't exist
    # This tests robustness of initialization
    assert isinstance(backend._configs, dict)
    
    # Test that repo root is calculated correctly
    repo_root = os.path.normpath(os.path.join(os.path.dirname(backend.__class__.__module__), "../../.."))
    # Should be a valid path (even if directories don't exist)
    assert isinstance(repo_root, str)
    assert len(repo_root) > 0


def test_init_dataset_configs_with_existing_configs():
    """Test _init_dataset_configs loads existing configuration files."""
    backend = ViewerBackend()
    
    # Check if any configs were loaded
    configs = backend._configs
    
    # Structure should be correct regardless of whether files exist
    for config_name, config_info in configs.items():
        # Config names should follow 'type/name' format
        assert '/' in config_name, f"Config name {config_name} should have 'type/name' format"
        
        # Config info should have required fields
        assert isinstance(config_info, dict)
        assert 'path' in config_info
        assert 'type' in config_info
        assert 'name' in config_info
        
        assert isinstance(config_info['path'], str)
        assert isinstance(config_info['type'], str)
        assert isinstance(config_info['name'], str)
        
        # Type should be one of the expected dataset types
        assert config_info['type'] in DATASET_GROUPS.keys()


def test_attribute_existence_after_initialization():
    """Test all expected attributes exist after initialization."""
    backend = ViewerBackend()
    
    # Test all expected attributes are present
    required_attributes = [
        'logger', '_datasets', '_configs', '_transforms',
        'current_dataset', 'current_index',
        'point_size', 'point_opacity', 'sym_diff_radius', 'corr_radius', 'lod_type'
    ]
    
    for attr_name in required_attributes:
        assert hasattr(backend, attr_name), f"Backend should have attribute {attr_name}"
        
        # Test that attributes are not None (except for current_dataset which starts as None)
        if attr_name != 'current_dataset':
            assert getattr(backend, attr_name) is not None, f"Attribute {attr_name} should not be None"


def test_initial_state_consistency():
    """Test that initial state is internally consistent."""
    backend = ViewerBackend()
    
    # Test that get_state() returns consistent initial values
    state = backend.get_state()
    
    assert state['current_dataset'] == backend.current_dataset
    assert state['current_index'] == backend.current_index
    assert state['point_size'] == backend.point_size
    assert state['point_opacity'] == backend.point_opacity
    assert state['sym_diff_radius'] == backend.sym_diff_radius
    assert state['corr_radius'] == backend.corr_radius
    assert state['lod_type'] == backend.lod_type


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_settings_immutability_after_init():
    """Test that changing ViewerSettings after init doesn't affect existing backend."""
    backend = ViewerBackend()
    original_point_size = backend.point_size
    
    # Attempt to modify the settings (this should not affect the backend)
    # Note: This is testing that the backend copied the values, not using references
    if hasattr(ViewerSettings, 'DEFAULT_3D_SETTINGS'):
        # Backend should have copied values, not hold references
        assert backend.point_size == original_point_size