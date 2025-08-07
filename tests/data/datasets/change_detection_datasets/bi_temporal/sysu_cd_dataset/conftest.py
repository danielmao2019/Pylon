"""Shared fixtures and helper functions for SYSU-CD dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.sysu_cd_dataset import SYSU_CD_Dataset


@pytest.fixture
def sysu_cd_dataset_train_config(sysu_cd_data_root, use_cpu_device, get_device):
    """Fixture for creating a SYSU_CD_Dataset config with train split."""
    return {
        'class': SYSU_CD_Dataset,
        'args': {
            'data_root': sysu_cd_data_root,
            'split': 'train',
            'device': get_device(use_cpu_device)
        }
    }


@pytest.fixture
def dataset_config(request, sysu_cd_data_root, use_cpu_device, get_device):
    """Fixture for creating a SYSU_CD_Dataset config with parameterized split."""
    split = request.param
    return {
        'class': SYSU_CD_Dataset,
        'args': {
            'data_root': sysu_cd_data_root,
            'split': split,
            'device': get_device(use_cpu_device)
        }
    }
