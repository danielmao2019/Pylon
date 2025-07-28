"""Shared fixtures and helper functions for AirChangeDataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset

# air_change_data_root fixture is now provided by parent conftest.py


@pytest.fixture
def air_change_dataset_train(air_change_data_root):
    """Fixture for creating an AirChangeDataset instance with train split."""
    return AirChangeDataset(data_root=air_change_data_root, split='train')


@pytest.fixture
def dataset(request, air_change_data_root):
    """Fixture for creating an AirChangeDataset instance with parameterized split."""
    split = request.param
    return AirChangeDataset(data_root=air_change_data_root, split=split)
