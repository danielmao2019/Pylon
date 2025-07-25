"""Shared fixtures for OSCD dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.oscd_dataset import OSCDDataset


@pytest.fixture
def oscd_dataset_train():
    """Fixture for creating an OSCDDataset instance with train split."""
    return OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='train')


@pytest.fixture  
def oscd_dataset_test():
    """Fixture for creating an OSCDDataset instance with test split."""
    return OSCDDataset(data_root="./data/datasets/soft_links/OSCD", split='test')
