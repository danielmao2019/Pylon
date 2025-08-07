"""Shared fixtures and helper functions for AirChangeDataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset


@pytest.fixture
def air_change_dataset_train_config(air_change_data_root):
    """Fixture for creating an AirChangeDataset config with train split."""
    return {
        'class': AirChangeDataset,
        'args': {
            'data_root': air_change_data_root,
            'split': 'train'
        }
    }


@pytest.fixture
def dataset_config(request, air_change_data_root):
    """Fixture for creating an AirChangeDataset config with parameterized split."""
    split = request.param
    return {
        'class': AirChangeDataset,
        'args': {
            'data_root': air_change_data_root,
            'split': split
        }
    }
