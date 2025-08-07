"""Shared fixtures and helper functions for OSCD dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset


@pytest.fixture
def oscd_dataset_train_config(oscd_data_root):
    """Fixture for creating an OSCDDataset config with train split."""
    return {
        'class': OSCDDataset,
        'args': {
            'data_root': oscd_data_root,
            'split': 'train'
        }
    }


@pytest.fixture
def oscd_dataset_test_config(oscd_data_root):
    """Fixture for creating an OSCDDataset config with test split."""
    return {
        'class': OSCDDataset,
        'args': {
            'data_root': oscd_data_root,
            'split': 'test'
        }
    }


@pytest.fixture
def dataset_config(request, oscd_data_root):
    """Fixture for creating an OSCDDataset config with parameterized split."""
    split = request.param
    return {
        'class': OSCDDataset,
        'args': {
            'data_root': oscd_data_root,
            'split': split
        }
    }
