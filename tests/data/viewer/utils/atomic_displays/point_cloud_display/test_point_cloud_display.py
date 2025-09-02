"""Tests for point cloud display functionality - Valid cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

import plotly.graph_objects as go
from dash import html

from data.viewer.utils.atomic_displays.point_cloud_display import (
    create_point_cloud_display
)


# ================================================================================
# Fixtures
# ================================================================================

@pytest.fixture
def point_cloud_3d():
    """Fixture providing 3D point cloud tensor."""
    return torch.randn(1000, 3, dtype=torch.float32)


@pytest.fixture
def point_cloud_colors():
    """Fixture providing point cloud colors."""
    return torch.randint(0, 255, (1000, 3), dtype=torch.uint8)


@pytest.fixture
def point_cloud_labels():
    """Fixture providing point cloud labels."""
    return torch.randint(0, 5, (1000,), dtype=torch.long)


# ================================================================================
# create_point_cloud_display Tests - Valid Cases
# ================================================================================

def test_create_point_cloud_display_basic(point_cloud_3d):
    """Test basic point cloud display creation."""
    fig = create_point_cloud_display(
        points=point_cloud_3d,
        colors=None,
        labels=None,
        highlight_indices=None,
        title="Test Point Cloud", 
        lod_type="none"
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Point Cloud"


def test_create_point_cloud_display_with_colors(point_cloud_3d, point_cloud_colors):
    """Test point cloud display with colors."""
    fig = create_point_cloud_display(
        points=point_cloud_3d,
        colors=point_cloud_colors,
        labels=None,
        highlight_indices=None,
        title="Colored Point Cloud",
        lod_type="none"
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Colored Point Cloud"


def test_create_point_cloud_display_with_labels(point_cloud_3d, point_cloud_labels):
    """Test point cloud display with labels."""
    fig = create_point_cloud_display(
        points=point_cloud_3d,
        colors=None,
        labels=point_cloud_labels,
        highlight_indices=None,
        title="Labeled Point Cloud",
        lod_type="none"
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Labeled Point Cloud"


def test_create_point_cloud_display_with_lod():
    """Test point cloud display with different LOD types."""
    points = torch.randn(1000, 3, dtype=torch.float32)
    camera_state = {'eye': {'x': 1, 'y': 1, 'z': 1}, 'center': {'x': 0, 'y': 0, 'z': 0}, 'up': {'x': 0, 'y': 0, 'z': 1}}
    
    # Test continuous LOD (needs camera_state)
    fig_continuous = create_point_cloud_display(
        points=points,
        colors=None,
        labels=None,
        highlight_indices=None,
        title="Continuous LOD",
        lod_type="continuous",
        camera_state=camera_state
    )
    assert isinstance(fig_continuous, go.Figure)
    
    # Test discrete LOD (needs point_cloud_id and camera_state)
    fig_discrete = create_point_cloud_display(
        points=points,
        colors=None,
        labels=None,
        highlight_indices=None,
        title="Discrete LOD",
        lod_type="discrete",
        point_cloud_id="test_id",
        camera_state=camera_state
    )
    assert isinstance(fig_discrete, go.Figure)
    
    # Test none LOD
    fig_none = create_point_cloud_display(
        points=points,
        colors=None,
        labels=None,
        highlight_indices=None,
        title="No LOD",
        lod_type="none"
    )
    assert isinstance(fig_none, go.Figure)


# ================================================================================
# Integration and Edge Case Tests
# ================================================================================

def test_point_cloud_display_pipeline(point_cloud_3d):
    """Test complete point cloud display pipeline."""
    # Create display
    fig = create_point_cloud_display(
        points=point_cloud_3d,
        colors=None,
        labels=None,
        highlight_indices=None,
        title="Pipeline Test",
        lod_type="none"
    )
    assert isinstance(fig, go.Figure)


def test_large_point_cloud_performance():
    """Test performance with large point clouds."""
    # Create large point cloud
    large_pc = torch.randn(10000, 3, dtype=torch.float32)
    
    # This should complete without error
    fig = create_point_cloud_display(
        points=large_pc,
        colors=None,
        labels=None,
        highlight_indices=None,
        title="Large PC Test",
        lod_type="none"
    )
    
    # Basic checks
    assert isinstance(fig, go.Figure)


def test_edge_case_point_clouds():
    """Test edge cases for point cloud processing."""
    # Very small coordinates
    tiny_pc = torch.full((100, 3), 1e-6, dtype=torch.float32)
    fig = create_point_cloud_display(
        points=tiny_pc,
        colors=None,
        labels=None,
        highlight_indices=None,
        title="Tiny PC",
        lod_type="none"
    )
    assert isinstance(fig, go.Figure)
    
    # Very large coordinates
    huge_pc = torch.full((100, 3), 1e6, dtype=torch.float32)
    fig = create_point_cloud_display(
        points=huge_pc,
        colors=None,
        labels=None,
        highlight_indices=None,
        title="Huge PC",
        lod_type="none"
    )
    assert isinstance(fig, go.Figure)
    
    # Mixed positive/negative
    mixed_pc = torch.randn(100, 3, dtype=torch.float32) * 1000
    fig = create_point_cloud_display(
        points=mixed_pc,
        colors=None,
        labels=None,
        highlight_indices=None,
        title="Mixed PC",
        lod_type="none"
    )
    assert isinstance(fig, go.Figure)
