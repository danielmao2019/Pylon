"""Shared fixtures and helper functions for cdd_dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.cdd_dataset import CDDDataset


@pytest.fixture
def cdd_dataset_train_config(cdd_data_root, use_cpu_device, get_device):
    """Fixture for creating a CDDDataset config with train split."""
    return {
        'class': CDDDataset,
        'args': {
            'data_root': cdd_data_root,
            'split': 'train',
            'device': get_device(use_cpu_device)
        }
    }


@pytest.fixture
def dataset_config(request, cdd_data_root, use_cpu_device, get_device):
    """Fixture for creating a CDDDataset config with parameterized split."""
    split = request.param
    return {
        'class': CDDDataset,
        'args': {
            'data_root': cdd_data_root,
            'split': split,
            'device': get_device(use_cpu_device)
        }
    }
