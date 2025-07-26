"""Shared fixtures and helper functions for AirChangeDataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.air_change_dataset import AirChangeDataset


@pytest.fixture
def air_change_data_root():
    """Fixture that returns the real AirChange dataset path."""
    return "./data/datasets/soft_links/AirChange"


@pytest.fixture
def air_change_dataset_train(air_change_data_root):
    """Fixture for creating an AirChangeDataset instance with train split."""
    return AirChangeDataset(data_root=air_change_data_root, split='train')


@pytest.fixture
def dataset(request, air_change_data_root):
    """Fixture for creating an AirChangeDataset instance with parameterized split."""
    split = request.param
    return AirChangeDataset(data_root=air_change_data_root, split=split)
