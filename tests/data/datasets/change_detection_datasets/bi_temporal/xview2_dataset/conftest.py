"""Shared fixtures and helper functions for xView2 dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import xView2Dataset


@pytest.fixture
def xview2_dataset_train_config(xview2_data_root):
    """Fixture for creating an xView2Dataset config with train split."""
    return {
        'class': xView2Dataset,
        'args': {
            'data_root': xview2_data_root,
            'split': 'train'
        }
    }


@pytest.fixture
def dataset_config(request, xview2_data_root):
    """Fixture for creating an xView2Dataset config with parameterized split."""
    split = request.param
    return {
        'class': xView2Dataset,
        'args': {
            'data_root': xview2_data_root,
            'split': split
        }
    }
