"""Tests for point cloud utility functions.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""

import pytest
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.viewer.utils.atomic_displays.point_cloud_display import (
    build_point_cloud_id,
    get_point_cloud_display_stats,
    normalize_point_cloud_id,
)

# ================================================================================
# Fixtures
# ================================================================================


@pytest.fixture
def point_cloud_3d():
    """Fixture providing 3D point cloud tensor."""
    return torch.randn(1000, 3, dtype=torch.float32)


@pytest.fixture
def sample_datapoint():
    """Fixture providing sample datapoint for ID building."""
    return {'meta_info': {'idx': 42}, 'other_data': 'test'}


@pytest.fixture
def setup_viewer_context():
    """Set up minimal viewer context for testing."""
    import data.viewer.dataset.context as viewer_context_module
    from data.viewer.dataset.backend import ViewerBackend
    from data.viewer.dataset.context import DatasetViewerContext, set_viewer_context

    backend = ViewerBackend()
    backend.current_dataset = 'test_dataset'
    context = DatasetViewerContext(backend=backend, available_datasets={})

    original_context = viewer_context_module._VIEWER_CONTEXT
    set_viewer_context(context)

    yield

    viewer_context_module._VIEWER_CONTEXT = original_context


# ================================================================================
# normalize_point_cloud_id Tests
# ================================================================================


def test_normalize_point_cloud_id_string():
    """Test that string IDs pass through unchanged."""
    point_cloud_id = "simple_id"
    result = normalize_point_cloud_id(point_cloud_id)

    assert result == "simple_id"


def test_normalize_point_cloud_id_tuple():
    """Test that tuple IDs are converted to colon-separated strings."""
    point_cloud_id = ("pcr/kitti", "42", "source")
    result = normalize_point_cloud_id(point_cloud_id)

    assert result == "pcr/kitti:42:source"


@pytest.mark.parametrize(
    "point_cloud_id,expected",
    [
        ("simple_id", "simple_id"),
        (("pcr/kitti", "42", "source"), "pcr/kitti:42:source"),
        (("change_detection", "10", "union"), "change_detection:10:union"),
        (("single",), "single"),
        (("a", "b", "c", "d"), "a:b:c:d"),
    ],
)
def test_normalize_point_cloud_id_various_inputs(point_cloud_id, expected):
    """Test normalize_point_cloud_id with various inputs."""
    result = normalize_point_cloud_id(point_cloud_id)
    assert result == expected


# ================================================================================
# build_point_cloud_id Tests - Require viewer system setup
# ================================================================================
# NOTE: These tests require the viewer context to be initialized
# They should be moved to integration tests or run with proper viewer setup


def test_build_point_cloud_id_basic(sample_datapoint, setup_viewer_context):
    """Test basic point cloud ID building."""
    result = build_point_cloud_id(sample_datapoint, "source")

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "test_dataset"  # From our MinimalBackend
    assert result[1] == 42  # From sample_datapoint meta_info
    assert result[2] == "source"


def test_build_point_cloud_id_different_components(
    sample_datapoint, setup_viewer_context
):
    """Test building IDs with different component names."""
    components = ["source", "target", "union", "intersection"]

    for component in components:
        result = build_point_cloud_id(sample_datapoint, component)
        assert result[2] == component


def test_build_point_cloud_id_missing_meta_info(setup_viewer_context):
    """Test behavior when meta_info is missing (should default to idx=0)."""
    datapoint = {"other_data": "test"}

    result = build_point_cloud_id(datapoint, "source")

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], str)  # dataset name from context
    assert result[1] == 0  # default idx when meta_info missing
    assert result[2] == "source"


def test_build_point_cloud_id_missing_idx_in_meta_info(setup_viewer_context):
    """Test behavior when idx is missing from meta_info (should default to 0)."""
    datapoint = {"meta_info": {"other_field": "value"}}

    result = build_point_cloud_id(datapoint, "source")

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], str)  # dataset name from context
    assert result[1] == 0  # default idx when not in meta_info
    assert result[2] == "source"


# ================================================================================
# get_point_cloud_display_stats Tests
# ================================================================================


def test_get_point_cloud_display_stats_basic():
    """Test basic point cloud statistics."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32
    )

    pc = PointCloud(xyz=points)
    stats = get_point_cloud_display_stats(pc)

    assert isinstance(stats, dict)
    assert stats['total_points'] == 3
    assert stats['dimensions'] == 3
    assert 'x_range' in stats
    assert 'y_range' in stats
    assert 'z_range' in stats
    assert 'center' in stats


def test_get_point_cloud_display_stats_basic_extended():
    """Test basic point cloud statistics with additional checks."""
    points = torch.randn(100, 3, dtype=torch.float32)

    pc = PointCloud(xyz=points)
    stats = get_point_cloud_display_stats(pc)

    assert isinstance(stats, dict)
    assert stats['total_points'] == 100
    assert stats['dimensions'] == 3
    assert 'x_range' in stats
    assert 'y_range' in stats
    assert 'z_range' in stats
    assert 'center' in stats


def test_get_point_cloud_display_stats_with_change_map():
    """Test point cloud statistics with change map."""
    points = torch.randn(100, 3, dtype=torch.float32)
    change_map = torch.randint(0, 3, (100,), dtype=torch.long)

    pc = PointCloud(xyz=points)
    stats = get_point_cloud_display_stats(pc, change_map=change_map)

    assert isinstance(stats, dict)
    assert stats['total_points'] == 100
    assert stats['dimensions'] == 3
    assert 'class_distribution' in stats
    assert isinstance(stats['class_distribution'], dict)


def test_get_point_cloud_display_stats_invalid_points_shape():
    """Test that invalid coordinate dimensions raise assertion."""
    points = torch.randn(100, 4, dtype=torch.float32)

    with pytest.raises(AssertionError):
        PointCloud(xyz=points)


# ================================================================================
# Integration Tests for Utility Functions
# ================================================================================


def test_point_cloud_id_pipeline(sample_datapoint, setup_viewer_context):
    """Test complete point cloud ID pipeline."""
    # Build ID
    pc_id = build_point_cloud_id(sample_datapoint, "source")
    assert isinstance(pc_id, tuple)
    assert len(pc_id) == 3

    # Normalize ID
    normalized = normalize_point_cloud_id(pc_id)
    assert isinstance(normalized, str)
    assert normalized == "test_dataset:42:source"

    # Test round-trip consistency
    assert normalize_point_cloud_id(normalized) == normalized


def test_point_cloud_utility_pipeline(point_cloud_3d):
    """Test complete point cloud utility pipeline."""
    # Get statistics
    pc = PointCloud(xyz=point_cloud_3d)
    stats = get_point_cloud_display_stats(pc)
    assert isinstance(stats, dict)
    assert stats['total_points'] == 1000
    assert stats['dimensions'] == 3
