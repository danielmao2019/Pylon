"""Shared fixtures and helper functions for Urb3DCD dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset


@pytest.fixture
def urb3dcd_dataset_train(urb3dcd_data_root):
    """Fixture for creating an Urb3DCDDataset instance with train split."""
    return Urb3DCDDataset(
        data_root=urb3dcd_data_root, 
        split='train',
        version=1,
        patched=True,
        sample_per_epoch=128,
        fix_samples=False,
        radius=50
    )


@pytest.fixture
def dataset(request, urb3dcd_data_root):
    """Fixture for creating an Urb3DCDDataset instance with parameterized split."""
    split = request.param
    return Urb3DCDDataset(
        data_root=urb3dcd_data_root, 
        split=split,
        version=1,
        patched=True,
        sample_per_epoch=128,
        fix_samples=False,
        radius=50
    )
