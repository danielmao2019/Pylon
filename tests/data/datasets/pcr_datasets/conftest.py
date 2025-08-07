"""Shared fixtures and helper functions for PCR dataset tests."""

import pytest
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset
from data.datasets.pcr_datasets.threedmatch_dataset import ThreeDMatchDataset, ThreeDLoMatchDataset


@pytest.fixture
def kitti_dataset_config(request, kitti_data_root):
    """Fixture for creating a KITTIDataset config."""
    split = request.param
    return {
        'class': KITTIDataset,
        'args': {
            'data_root': kitti_data_root,
            'split': split
        }
    }


@pytest.fixture
def threedmatch_dataset_config(request, threedmatch_data_root):
    """Fixture for creating a ThreeDMatchDataset config."""
    split = request.param
    return {
        'class': ThreeDMatchDataset,
        'args': {
            'data_root': threedmatch_data_root,
            'split': split,
            'matching_radius': 0.1
        }
    }


@pytest.fixture
def threedlomatch_dataset_config(request, threedmatch_data_root):
    """Fixture for creating a ThreeDLoMatchDataset config."""
    split = request.param
    return {
        'class': ThreeDLoMatchDataset,
        'args': {
            'data_root': threedmatch_data_root,
            'split': split,
            'matching_radius': 0.1
        }
    }
