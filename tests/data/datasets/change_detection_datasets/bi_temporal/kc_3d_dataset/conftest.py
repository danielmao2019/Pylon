"""Shared fixtures and helper functions for KC-3D dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.kc_3d_dataset import KC3DDataset


@pytest.fixture
def kc_3d_dataset_train(kc_3d_data_root):
    """Fixture for creating a KC3DDataset instance with train split."""
    return KC3DDataset(data_root=kc_3d_data_root, split='train')


@pytest.fixture
def dataset(request, kc_3d_data_root):
    """Fixture for creating a KC3DDataset instance with parameterized split."""
    split = request.param
    return KC3DDataset(data_root=kc_3d_data_root, split=split)
