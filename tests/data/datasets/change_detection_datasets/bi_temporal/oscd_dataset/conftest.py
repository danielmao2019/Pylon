"""Shared fixtures and helper functions for OSCD dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset


@pytest.fixture
def oscd_dataset_train(oscd_data_root):
    """Fixture for creating an OSCDDataset instance with train split."""
    return OSCDDataset(data_root=oscd_data_root, split='train')


@pytest.fixture
def oscd_dataset_test(oscd_data_root):
    """Fixture for creating an OSCDDataset instance with test split."""
    return OSCDDataset(data_root=oscd_data_root, split='test')


@pytest.fixture
def dataset(request, oscd_data_root):
    """Fixture for creating an OSCDDataset instance with parameterized split."""
    split = request.param
    return OSCDDataset(data_root=oscd_data_root, split=split)
