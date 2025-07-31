"""Shared fixtures and helper functions for xView2Dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.xview2_dataset import xView2Dataset


@pytest.fixture
def xview2_data_root():
    """Fixture that returns the real xView2 dataset path."""
    return "./data/datasets/soft_links/xView2"


@pytest.fixture
def xview2_dataset_train(xview2_data_root):
    """Fixture for creating an xView2Dataset instance with train split."""
    return xView2Dataset(data_root=xview2_data_root, split='train')


@pytest.fixture
def dataset(request, xview2_data_root):
    """Fixture for creating an xView2Dataset instance with parameterized split."""
    split = request.param
    return xView2Dataset(data_root=xview2_data_root, split=split)
