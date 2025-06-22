"""Pytest fixtures for change detection model tests."""
import pytest
from tests.models.utils.change_detection_data import (
    generate_change_detection_data,
    generate_change_labels,
    generate_segmentation_labels,
    create_minimal_change_detection_input,
    create_large_change_detection_input
)


@pytest.fixture
def change_detection_data_generator():
    """Fixture that provides the change detection data generator function."""
    return generate_change_detection_data


@pytest.fixture
def change_labels_generator():
    """Fixture that provides the change labels generator function."""
    return generate_change_labels


@pytest.fixture
def segmentation_labels_generator():
    """Fixture that provides the segmentation labels generator function."""
    return generate_segmentation_labels


@pytest.fixture
def minimal_change_detection_input():
    """Fixture that provides minimal change detection input."""
    return create_minimal_change_detection_input


@pytest.fixture
def large_change_detection_input():
    """Fixture that provides large change detection input for memory testing."""
    return create_large_change_detection_input
