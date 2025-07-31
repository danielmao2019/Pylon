"""Shared fixtures and helper functions for LevirCdDataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset


@pytest.fixture
def levir_cd_data_root():
    """Fixture that returns the real LEVIR-CD dataset path."""
    return "./data/datasets/soft_links/LEVIR-CD"


@pytest.fixture
def levir_cd_dataset_train(levir_cd_data_root):
    """Fixture for creating a LevirCdDataset instance with train split."""
    return LevirCdDataset(data_root=levir_cd_data_root, split='train')


@pytest.fixture
def dataset(request, levir_cd_data_root):
    """Fixture for creating a LevirCdDataset instance with parameterized split."""
    split = request.param
    return LevirCdDataset(data_root=levir_cd_data_root, split=split)
