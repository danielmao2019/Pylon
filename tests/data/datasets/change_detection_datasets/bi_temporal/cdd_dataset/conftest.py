"""Shared fixtures and helper functions for cdd_dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.cdd_dataset import CDDDataset


@pytest.fixture
def cdd_dataset_train(cdd_data_root):
    """Fixture for creating a CDDDataset instance with train split."""
    return CDDDataset(data_root=cdd_data_root, split='train')


@pytest.fixture
def dataset(request, cdd_data_root):
    """Fixture for creating a CDDDataset instance with parameterized split."""
    split = request.param
    return CDDDataset(data_root=cdd_data_root, split=split)
