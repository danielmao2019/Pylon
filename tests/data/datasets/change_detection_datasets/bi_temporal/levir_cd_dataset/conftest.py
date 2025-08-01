"""Shared fixtures and helper functions for LEVIR-CD dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LEVIR_CD_Dataset


@pytest.fixture
def levir_cd_dataset_train(levir_cd_data_root):
    """Fixture for creating a LEVIR_CD_Dataset instance with train split."""
    return LEVIR_CD_Dataset(data_root=levir_cd_data_root, split='train')


@pytest.fixture
def dataset(request, levir_cd_data_root):
    """Fixture for creating a LEVIR_CD_Dataset instance with parameterized split."""
    split = request.param
    return LEVIR_CD_Dataset(data_root=levir_cd_data_root, split=split)
