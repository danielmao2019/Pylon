"""Shared fixtures and helper functions for SLPCCD dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset


@pytest.fixture
def slpccd_dataset_train_config(slpccd_data_root):
    """Fixture for creating a SLPCCDDataset config with train split."""
    return {
        'class': SLPCCDDataset,
        'args': {
            'data_root': slpccd_data_root,
            'split': 'train',
            'num_points': 8192,
            'random_subsample': True,
            'use_hierarchy': True,
            'hierarchy_levels': 3,
            'knn_size': 16,
            'cross_knn_size': 16
        }
    }


@pytest.fixture
def dataset_config(request, slpccd_data_root):
    """Fixture for creating a SLPCCDDataset config with parameterized split."""
    split = request.param
    return {
        'class': SLPCCDDataset,
        'args': {
            'data_root': slpccd_data_root,
            'split': split,
            'num_points': 8192,
            'random_subsample': True,
            'use_hierarchy': True,
            'hierarchy_levels': 3,
            'knn_size': 16,
            'cross_knn_size': 16
        }
    }
