"""Shared fixtures and helper functions for multi-task dataset tests."""

import pytest
from data.datasets import ADE20KDataset
from data.datasets.multi_task_datasets.celeb_a_dataset import CelebADataset
from data.datasets.multi_task_datasets.city_scapes_dataset import CityScapesDataset
from data.datasets.multi_task_datasets.multi_task_facial_landmark_dataset import MultiTaskFacialLandmarkDataset
from data.datasets.multi_task_datasets.nyu_v2_dataset import NYUv2Dataset
from data.datasets.multi_task_datasets.pascal_context_dataset import PASCALContextDataset


@pytest.fixture
def celeb_a_dataset_config(request, celeb_a_data_root):
    """Fixture for creating a CelebADataset config."""
    params = request.param
    return {
        'class': CelebADataset,
        'args': {
            'data_root': celeb_a_data_root,
            **params
        }
    }


@pytest.fixture
def ade20k_dataset_config(request, ade20k_data_root):
    """Fixture for creating an ADE20KDataset config."""
    split = request.param
    return {
        'class': ADE20KDataset,
        'args': {
            'data_root': ade20k_data_root,
            'split': split
        }
    }


@pytest.fixture
def city_scapes_dataset_config(request, city_scapes_data_root):
    """Fixture for creating a CityScapesDataset config."""
    params = request.param
    return {
        'class': CityScapesDataset,
        'args': {
            'data_root': city_scapes_data_root,
            **params
        }
    }


@pytest.fixture
def city_scapes_base_config(city_scapes_data_root):
    """Base config for CityScapes selective loading tests."""
    return {
        'class': CityScapesDataset,
        'args': {
            'data_root': city_scapes_data_root,
            'split': 'train'
        }
    }


@pytest.fixture
def multi_task_facial_landmark_dataset_config(request, multi_task_facial_landmark_data_root):
    """Fixture for creating a MultiTaskFacialLandmarkDataset config."""
    params = request.param
    return {
        'class': MultiTaskFacialLandmarkDataset,
        'args': {
            'data_root': multi_task_facial_landmark_data_root,
            **params
        }
    }


@pytest.fixture
def nyu_v2_dataset_config(request, nyu_v2_data_root):
    """Fixture for creating a NYUv2Dataset config."""
    params = request.param
    return {
        'class': NYUv2Dataset,
        'args': {
            'data_root': nyu_v2_data_root,
            **params
        }
    }


@pytest.fixture
def nyu_v2_base_config(nyu_v2_data_root):
    """Base config for NYUv2 selective loading tests."""
    return {
        'class': NYUv2Dataset,
        'args': {
            'data_root': nyu_v2_data_root,
            'split': 'train'
        }
    }


@pytest.fixture
def pascal_context_dataset_config(request, pascal_context_data_root):
    """Fixture for creating a PASCALContextDataset config."""
    split = request.param
    return {
        'class': PASCALContextDataset,
        'args': {
            'data_root': pascal_context_data_root,
            'split': split
        }
    }


@pytest.fixture
def pascal_context_base_config(pascal_context_data_root):
    """Base config for PASCAL Context selective loading tests."""
    return {
        'class': PASCALContextDataset,
        'args': {
            'data_root': pascal_context_data_root,
            'split': 'train'
        }
    }
