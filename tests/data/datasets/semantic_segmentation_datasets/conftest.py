"""Shared fixtures and helper functions for semantic segmentation dataset tests."""

import pytest
from data.datasets.semantic_segmentation_datasets.whu_bd_dataset import WHU_BD_Dataset


@pytest.fixture
def whu_bd_dataset_config(request, whu_bd_data_root):
    """Fixture for creating a WHU_BD_Dataset config."""
    split = request.param
    return {
        'class': WHU_BD_Dataset,
        'args': {
            'data_root': whu_bd_data_root,
            'split': split
        }
    }
