"""Shared fixtures and helper functions for semantic segmentation dataset tests."""

import pytest
from data.datasets.semantic_segmentation_datasets.whu_bd_dataset import WHU_BD_Dataset
from data.datasets.semantic_segmentation_datasets.coco_stuff_164k_dataset import COCOStuff164KDataset


@pytest.fixture
def whu_bd_dataset_config(request, whu_bd_data_root, use_cpu_device, get_device):
    """Fixture for creating a WHU_BD_Dataset config."""
    split = request.param
    return {
        'class': WHU_BD_Dataset,
        'args': {
            'data_root': whu_bd_data_root,
            'split': split,
            'device': get_device(use_cpu_device)
        }
    }


@pytest.fixture
def dataset_config(request, coco_stuff_164k_data_root, use_cpu_device, get_device):
    """Fixture for creating a COCOStuff164KDataset config."""
    split, semantic_granularity = request.param
    return {
        'class': COCOStuff164KDataset,
        'args': {
            'data_root': coco_stuff_164k_data_root,
            'split': split,
            'semantic_granularity': semantic_granularity,
            'device': get_device(use_cpu_device)
        }
    }
