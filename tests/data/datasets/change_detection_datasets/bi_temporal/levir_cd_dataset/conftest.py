"""Shared fixtures and helper functions for LEVIR-CD dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import LevirCdDataset


@pytest.fixture
def levir_cd_dataset_train_config(levir_cd_data_root):
    """Fixture for creating a LevirCdDataset config with train split."""
    return {
        'class': LevirCdDataset,
        'args': {
            'data_root': levir_cd_data_root,
            'split': 'train'
        }
    }


@pytest.fixture
def dataset_config(request, levir_cd_data_root):
    """Fixture for creating a LevirCdDataset config with parameterized split."""
    split = request.param
    return {
        'class': LevirCdDataset,
        'args': {
            'data_root': levir_cd_data_root,
            'split': split
        }
    }
