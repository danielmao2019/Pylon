"""Tests for ViewerBackend state management functionality.

This module tests state updates, retrieval, and consistency of the ViewerBackend class.
"""

from typing import Dict, Any

import pytest

from data.viewer.backend.backend import ViewerBackend


@pytest.fixture
def backend():
    """Create a ViewerBackend instance for testing."""
    return ViewerBackend()


def test_get_state_returns_complete_state(backend):
    """Test that get_state returns all expected state fields."""
    state = backend.get_state()

    # Test state structure
    assert isinstance(state, dict)

    # Test all expected keys are present
    expected_keys = [
        'current_dataset', 'current_index', 'point_size', 'point_opacity',
        'sym_diff_radius', 'corr_radius', 'lod_type'
    ]
    for key in expected_keys:
        assert key in state, f"State should contain key: {key}"

    # Test initial state values
    assert state['current_dataset'] is None
    assert state['current_index'] == 0
    assert isinstance(state['point_size'], float)
    assert isinstance(state['point_opacity'], float)
    assert isinstance(state['sym_diff_radius'], float)
    assert isinstance(state['corr_radius'], float)
    assert isinstance(state['lod_type'], str)


def test_update_state_single_value(backend):
    """Test updating a single state value."""
    # Test updating current_index
    backend.update_state(current_index=5)
    assert backend.current_index == 5
    assert backend.get_state()['current_index'] == 5

    # Test updating point_size
    backend.update_state(point_size=10.0)
    assert backend.point_size == 10.0
    assert backend.get_state()['point_size'] == 10.0

    # Test updating point_opacity
    backend.update_state(point_opacity=0.8)
    assert backend.point_opacity == 0.8
    assert backend.get_state()['point_opacity'] == 0.8


def test_update_state_multiple_values(backend):
    """Test updating multiple state values simultaneously."""
    updates = {
        'current_index': 10,
        'point_size': 15.0,
        'point_opacity': 0.9,
        'sym_diff_radius': 0.05,
        'corr_radius': 0.15,
        'lod_type': 'ADAPTIVE'
    }

    backend.update_state(**updates)

    # Verify all updates were applied
    for key, expected_value in updates.items():
        assert getattr(backend, key) == expected_value, f"Failed to update {key}"

    # Verify get_state reflects the changes
    state = backend.get_state()
    for key, expected_value in updates.items():
        assert state[key] == expected_value, f"get_state() doesn't reflect update to {key}"


def test_update_state_with_dataset_name(backend):
    """Test updating current_dataset state."""
    # Set current_dataset to a test value
    test_dataset_name = "test/example_dataset"
    backend.update_state(current_dataset=test_dataset_name)

    assert backend.current_dataset == test_dataset_name
    assert backend.get_state()['current_dataset'] == test_dataset_name

    # Set back to None
    backend.update_state(current_dataset=None)
    assert backend.current_dataset is None
    assert backend.get_state()['current_dataset'] is None


@pytest.mark.parametrize("point_size", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_update_point_size_values(backend, point_size):
    """Test updating point_size with various valid values."""
    backend.update_state(point_size=point_size)
    assert backend.point_size == point_size
    assert backend.get_state()['point_size'] == point_size


@pytest.mark.parametrize("point_opacity", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_update_point_opacity_values(backend, point_opacity):
    """Test updating point_opacity with various valid values."""
    backend.update_state(point_opacity=point_opacity)
    assert backend.point_opacity == point_opacity
    assert backend.get_state()['point_opacity'] == point_opacity


@pytest.mark.parametrize("radius", [0.01, 0.05, 0.1, 0.2, 0.5])
def test_update_radius_values(backend, radius):
    """Test updating radius values with various valid values."""
    backend.update_state(sym_diff_radius=radius, corr_radius=radius)
    assert backend.sym_diff_radius == radius
    assert backend.corr_radius == radius

    state = backend.get_state()
    assert state['sym_diff_radius'] == radius
    assert state['corr_radius'] == radius


@pytest.mark.parametrize("lod_type", ['FIXED', 'ADAPTIVE', 'SMART'])
def test_update_lod_type_values(backend, lod_type):
    """Test updating lod_type with various valid values."""
    backend.update_state(lod_type=lod_type)
    assert backend.lod_type == lod_type
    assert backend.get_state()['lod_type'] == lod_type


def test_update_state_preserves_other_values(backend):
    """Test that updating one value doesn't affect others."""
    # Set initial values
    initial_updates = {
        'current_index': 5,
        'point_size': 3.0,
        'point_opacity': 0.7,
        'sym_diff_radius': 0.08,
        'corr_radius': 0.12,
        'lod_type': 'ADAPTIVE'
    }
    backend.update_state(**initial_updates)

    # Update only one value
    backend.update_state(point_size=8.0)

    # Verify the updated value
    assert backend.point_size == 8.0

    # Verify other values are preserved
    assert backend.current_index == 5
    assert backend.point_opacity == 0.7
    assert backend.sym_diff_radius == 0.08
    assert backend.corr_radius == 0.12
    assert backend.lod_type == 'ADAPTIVE'


def test_state_consistency_across_calls(backend):
    """Test that state remains consistent across multiple get_state calls."""
    # Set some state
    backend.update_state(
        current_index=20,
        point_size=4.5,
        point_opacity=0.6
    )

    # Get state multiple times
    state1 = backend.get_state()
    state2 = backend.get_state()
    state3 = backend.get_state()

    # All calls should return identical dictionaries
    assert state1 == state2 == state3

    # Values should match backend attributes
    assert state1['current_index'] == backend.current_index
    assert state1['point_size'] == backend.point_size
    assert state1['point_opacity'] == backend.point_opacity


def test_state_independence_from_returned_dict(backend):
    """Test that modifying returned state dict doesn't affect backend state."""
    original_state = backend.get_state()
    original_index = original_state['current_index']

    # Modify the returned dictionary
    original_state['current_index'] = 999

    # Backend state should be unchanged
    assert backend.current_index == original_index
    assert backend.get_state()['current_index'] == original_index


def test_update_state_with_mixed_types(backend):
    """Test updating state with mixed data types."""
    backend.update_state(
        current_dataset="mixed/test",
        current_index=42,
        point_size=7.5,
        point_opacity=0.85,
        sym_diff_radius=0.03,
        corr_radius=0.18,
        lod_type="FIXED"
    )

    # Verify types are preserved
    assert isinstance(backend.current_dataset, str)
    assert isinstance(backend.current_index, int)
    assert isinstance(backend.point_size, float)
    assert isinstance(backend.point_opacity, float)
    assert isinstance(backend.sym_diff_radius, float)
    assert isinstance(backend.corr_radius, float)
    assert isinstance(backend.lod_type, str)


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_update_state_ignores_nonexistent_attributes(backend):
    """Test that update_state ignores attributes that don't exist."""
    # This should not raise an error, but should not update anything
    initial_state = backend.get_state()

    backend.update_state(
        nonexistent_attribute="should_be_ignored",
        another_fake_attr=123
    )

    # State should be unchanged
    final_state = backend.get_state()
    assert initial_state == final_state

    # Backend should not have the nonexistent attributes
    assert not hasattr(backend, 'nonexistent_attribute')
    assert not hasattr(backend, 'another_fake_attr')


def test_update_state_with_empty_kwargs(backend):
    """Test that update_state with no arguments works correctly."""
    initial_state = backend.get_state()

    # This should not change anything
    backend.update_state()

    final_state = backend.get_state()
    assert initial_state == final_state