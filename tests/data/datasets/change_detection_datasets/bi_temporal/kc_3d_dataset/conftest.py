"""Shared fixtures and helper functions for KC-3D dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.kc_3d_dataset import KC3DDataset


@pytest.fixture
def kc_3d_dataset_train_config(kc_3d_data_root):
    """Fixture for creating a KC3DDataset config with train split."""
    return {
        'class': KC3DDataset,
        'args': {
            'data_root': kc_3d_data_root,
            'split': 'train'
        }
    }


@pytest.fixture
def dataset_config(request, kc_3d_data_root):
    """Fixture for creating a KC3DDataset config with parameterized split."""
    split = request.param
    return {
        'class': KC3DDataset,
        'args': {
            'data_root': kc_3d_data_root,
            'split': split
        }
    }
