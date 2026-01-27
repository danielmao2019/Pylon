"""Tests for point cloud ID utilities.

Focuses specifically on build_point_cloud_id and normalize_point_cloud_id functions.
CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""

import pytest

from data.viewer.utils.atomic_displays.point_cloud_display import (
    build_point_cloud_id,
    normalize_point_cloud_id,
)


@pytest.fixture(autouse=True)
def setup_viewer_context():
    """Set up viewer context for point cloud ID tests."""
    import data.viewer.context as viewer_context_module
    from data.viewer.backend import ViewerBackend
    from data.viewer.context import DatasetViewerContext, set_viewer_context

    backend = ViewerBackend()
    backend.current_dataset = "test_dataset"
    context = DatasetViewerContext(backend=backend, available_datasets={})

    original_context = viewer_context_module._VIEWER_CONTEXT
    set_viewer_context(context)

    yield

    viewer_context_module._VIEWER_CONTEXT = original_context


# ================================================================================
# build_point_cloud_id Tests
# ================================================================================


def test_build_point_cloud_id_basic():
    """Test basic point cloud ID generation."""
    datapoint = {"meta_info": {"idx": 42}}
    component = "source"

    pc_id = build_point_cloud_id(datapoint, component)

    assert isinstance(pc_id, tuple)
    assert len(pc_id) == 3
    assert pc_id[0] == "test_dataset"
    assert pc_id[1] == 42
    assert pc_id[2] == "source"


def test_build_point_cloud_id_deterministic():
    """Test that point cloud ID generation is deterministic."""
    datapoint = {"meta_info": {"idx": 10}}
    component = "target"

    pc_id1 = build_point_cloud_id(datapoint, component)
    pc_id2 = build_point_cloud_id(datapoint, component)

    assert isinstance(pc_id1, tuple)
    assert isinstance(pc_id2, tuple)
    assert pc_id1 == pc_id2  # Should be identical for same input


def test_build_point_cloud_id_different_inputs():
    """Test that different point clouds generate different IDs."""
    datapoint1 = {"meta_info": {"idx": 42}}
    datapoint2 = {"meta_info": {"idx": 123}}

    id1 = build_point_cloud_id(datapoint1, "source")
    id2 = build_point_cloud_id(datapoint2, "source")

    assert isinstance(id1, tuple)
    assert isinstance(id2, tuple)
    assert len(id1) == 3
    assert len(id2) == 3
    assert id1 != id2  # Different inputs should generate different IDs


def test_build_point_cloud_id_various_sizes():
    """Test point cloud ID generation with various datapoint indices."""
    indices = [1, 10, 100, 1000]
    ids = []

    for idx in indices:
        datapoint = {"meta_info": {"idx": idx}}
        pc_id = build_point_cloud_id(datapoint, "source")

        assert isinstance(pc_id, tuple)
        assert len(pc_id) == 3
        assert pc_id[1] == idx  # Check the index is correct
        ids.append(pc_id)

    # All IDs should be different
    assert len(set(ids)) == len(ids)


def test_build_point_cloud_id_single_point():
    """Test point cloud ID generation with single datapoint."""
    datapoint = {"meta_info": {"idx": 1}}
    pc_id = build_point_cloud_id(datapoint, "target")

    assert isinstance(pc_id, tuple)
    assert len(pc_id) == 3
    assert pc_id[1] == 1
    assert pc_id[2] == "target"


def test_build_point_cloud_id_different_components():
    """Test point cloud ID generation with different components."""
    datapoint = {"meta_info": {"idx": 100}}

    # Different components
    id_source = build_point_cloud_id(datapoint, "source")
    id_target = build_point_cloud_id(datapoint, "target")
    id_change = build_point_cloud_id(datapoint, "change_map")

    assert isinstance(id_source, tuple)
    assert isinstance(id_target, tuple)
    assert isinstance(id_change, tuple)

    # Same datapoint index, different components
    assert id_source[1] == id_target[1] == id_change[1] == 100
    assert id_source[2] == "source"
    assert id_target[2] == "target"
    assert id_change[2] == "change_map"

    # All IDs should be different
    assert id_source != id_target != id_change


# ================================================================================
# normalize_point_cloud_id Tests
# ================================================================================


def test_normalize_point_cloud_id_basic():
    """Test basic point cloud ID normalization."""
    original_id = "my_point_cloud_123"
    normalized_id = normalize_point_cloud_id(original_id)

    assert isinstance(normalized_id, str)
    assert len(normalized_id) > 0


def test_normalize_point_cloud_id_various_inputs():
    """Test normalization with various input strings."""
    test_ids = [
        "simple_id",
        "point_cloud_with_numbers_123",
        "UPPERCASE_ID",
        "Mixed_Case_ID_456",
        "id-with-dashes",
        "id.with.dots",
        "id_with_underscores_789",
    ]

    for original_id in test_ids:
        normalized_id = normalize_point_cloud_id(original_id)

        assert isinstance(normalized_id, str)
        assert len(normalized_id) > 0


def test_normalize_point_cloud_id_special_characters():
    """Test normalization with special characters."""
    special_id = "point@cloud#with$special%chars!"
    normalized_id = normalize_point_cloud_id(special_id)

    assert isinstance(normalized_id, str)
    assert len(normalized_id) > 0


def test_normalize_point_cloud_id_long_string():
    """Test normalization with very long string."""
    long_id = "very_long_point_cloud_identifier_" * 10  # 340+ characters
    normalized_id = normalize_point_cloud_id(long_id)

    assert isinstance(normalized_id, str)
    assert len(normalized_id) > 0


def test_normalize_point_cloud_id_deterministic():
    """Test that normalization is deterministic."""
    original_id = "test_point_cloud_id"

    normalized_1 = normalize_point_cloud_id(original_id)
    normalized_2 = normalize_point_cloud_id(original_id)

    assert normalized_1 == normalized_2
