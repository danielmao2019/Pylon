"""Shared fixtures and helper functions for single temporal dataset tests."""

import pytest

from data.datasets import Bi2SingleTemporal, SYSU_CD_Dataset, WHU_BD_Dataset
from data.datasets.change_detection_datasets.single_temporal.i3pe_dataset import (
    I3PEDataset,
)
from data.datasets.change_detection_datasets.single_temporal.ppsl_dataset import (
    PPSLDataset,
)
from utils.builders.builder import build_from_config


@pytest.fixture
def ppsl_dataset_config(whu_bd_data_root, use_cpu_device, get_device):
    """Fixture for creating a PPSLDataset config."""
    source_config = {
        'class': WHU_BD_Dataset,
        'args': {
            'data_root': whu_bd_data_root,
            'split': 'train',
            'device': get_device(use_cpu_device)
        }
    }

    # Need to build source dataset to get its length for PPSLDataset
    source = build_from_config(source_config)

    return {
        'class': PPSLDataset,
        'args': {
            'source': source,
            'dataset_size': len(source)
        }
    }


@pytest.fixture
def i3pe_dataset_config(sysu_cd_data_root, use_cpu_device, get_device):
    """Fixture for creating an I3PEDataset config."""
    # Create source configs and build nested source datasets
    sysu_config = {
        'class': SYSU_CD_Dataset,
        'args': {
            'data_root': sysu_cd_data_root,
            'split': 'train',
            'device': get_device(use_cpu_device)
        }
    }
    sysu_dataset = build_from_config(sysu_config)

    bi2single_config = {
        'class': Bi2SingleTemporal,
        'args': {
            'source': sysu_dataset
        }
    }
    source = build_from_config(bi2single_config)

    return {
        'class': I3PEDataset,
        'args': {
            'source': source,
            'dataset_size': len(source),
            'exchange_ratio': 0.75
        }
    }
