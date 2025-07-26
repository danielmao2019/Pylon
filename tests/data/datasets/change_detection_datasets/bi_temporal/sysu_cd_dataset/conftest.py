"""Shared fixtures and helper functions for SYSU_CD_Dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.sysu_cd_dataset import SYSU_CD_Dataset


@pytest.fixture
def sysu_cd_data_root():
    """Fixture that returns the real SYSU-CD dataset path."""
    return "./data/datasets/soft_links/SYSU-CD"


@pytest.fixture
def sysu_cd_dataset_train(sysu_cd_data_root):
    """Fixture for creating a SYSU_CD_Dataset instance with train split."""
    return SYSU_CD_Dataset(data_root=sysu_cd_data_root, split='train')


@pytest.fixture
def dataset(request, sysu_cd_data_root):
    """Fixture for creating a SYSU_CD_Dataset instance with parameterized split."""
    split = request.param
    return SYSU_CD_Dataset(data_root=sysu_cd_data_root, split=split)