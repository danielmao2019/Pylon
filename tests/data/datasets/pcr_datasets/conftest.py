"""Shared fixtures and helper functions for PCR dataset tests."""

import pytest
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset
from data.datasets.pcr_datasets.threedmatch_dataset import ThreeDMatchDataset, ThreeDLoMatchDataset
from data.datasets.pcr_datasets.modelnet40_dataset import ModelNet40Dataset


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


@pytest.fixture
def modelnet40_dataset_config(request, modelnet40_data_root, modelnet40_cache_file):
    """Fixture for creating a ModelNet40Dataset config."""
    dataset_params = request.param.copy()
    return {
        'class': ModelNet40Dataset,
        'args': {
            'data_root': modelnet40_data_root,
            'cache_filepath': modelnet40_cache_file,
            **dataset_params
        }
    }
