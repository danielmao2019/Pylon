"""Shared fixtures and helper functions for urb3dcd_dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.urb3dcd_dataset import Urb3DCDDataset


@pytest.fixture
def urb3dcd_data_root():
    """Fixture that returns the real Urb3DCD dataset path."""
    return "./data/datasets/soft_links/Urb3DCD"


@pytest.fixture
def urb3dcd_dataset_train(urb3dcd_data_root):
    """Fixture for creating a Urb3DCDDataset instance with train configuration."""
    return Urb3DCDDataset(
        data_root=urb3dcd_data_root,
        sample_per_epoch=100,
        radius=100,
        fix_samples=False
    )


@pytest.fixture
def dataset(request, urb3dcd_data_root):
    """Fixture for creating a Urb3DCDDataset instance with parameterized settings."""
    sample_per_epoch, radius, fix_samples = request.param
    return Urb3DCDDataset(
        data_root=urb3dcd_data_root,
        sample_per_epoch=sample_per_epoch,
        radius=radius,
        fix_samples=fix_samples
    )
