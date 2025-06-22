"""Pytest fixtures for point cloud registration model tests."""
import pytest
from tests.models.utils.point_cloud_data import (
    generate_point_cloud_data,
    generate_point_cloud_segmentation_data,
    generate_point_cloud_segmentation_labels,
    generate_transformation_matrix,
    create_minimal_point_cloud_input,
    create_large_point_cloud_input
)


@pytest.fixture
def point_cloud_data_generator():
    """Fixture that provides the point cloud data generator function."""
    return generate_point_cloud_data


@pytest.fixture
def point_cloud_segmentation_data_generator():
    """Fixture that provides the point cloud segmentation data generator function."""
    return generate_point_cloud_segmentation_data


@pytest.fixture
def point_cloud_segmentation_labels_generator():
    """Fixture that provides the point cloud segmentation labels generator function."""
    return generate_point_cloud_segmentation_labels


@pytest.fixture
def transformation_matrix_generator():
    """Fixture that provides the transformation matrix generator function."""
    return generate_transformation_matrix


@pytest.fixture
def minimal_point_cloud_input():
    """Fixture that provides minimal point cloud input."""
    return create_minimal_point_cloud_input


@pytest.fixture
def large_point_cloud_input():
    """Fixture that provides large point cloud input for memory testing."""
    return create_large_point_cloud_input
