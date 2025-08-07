"""Shared fixtures and helper functions for Urb3DCD dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset


@pytest.fixture
def urb3dcd_dataset_train_config(urb3dcd_data_root, use_cpu_device, get_device):
    """Fixture for creating an Urb3DCDDataset config with train split."""
    return {
        'class': Urb3DCDDataset,
        'args': {
            'data_root': urb3dcd_data_root,
            'split': 'train',
            'version': 1,
            'patched': True,
            'sample_per_epoch': 128,
            'fix_samples': False,
            'radius': 50,
            'device': get_device(use_cpu_device)
        }
    }


@pytest.fixture
def dataset_config(request, urb3dcd_data_root, use_cpu_device, get_device):
    """Fixture for creating an Urb3DCDDataset config with parameterized split."""
    split = request.param
    return {
        'class': Urb3DCDDataset,
        'args': {
            'data_root': urb3dcd_data_root,
            'split': split,
            'version': 1,
            'patched': True,
            'sample_per_epoch': 128,
            'fix_samples': False,
            'radius': 50,
            'device': get_device(use_cpu_device)
        }
    }
