"""Shared fixtures and helper functions for BaseDataset tests."""

import pytest
from typing import Dict, Any, Tuple
import torch
from data.datasets.base_dataset import BaseDataset


class MockDataset(BaseDataset):
    """Mock dataset for testing BaseDataset functionality."""

    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': 100, 'val': 20, 'test': 30}  # Class attribute for predefined splits
    INPUT_NAMES = ['input']
    LABEL_NAMES = ['label']
    SHA1SUM = None

    def __init__(self, data_root=None, split=None, split_percentages=None, **kwargs):
        # This dataset has predefined splits, so split=None is not allowed
        if split is None:
            raise ValueError("MockDataset has predefined splits - split=None is not allowed")

        super().__init__(data_root=data_root, split=split, split_percentages=split_percentages, **kwargs)

    def _init_annotations(self) -> None:
        # BaseDataset split logic:
        # 1. If split_percentages provided: load ALL data, BaseDataset will apply split after
        # 2. If no split_percentages: load only the specific split's data

        if hasattr(self, 'split_percentages') and self.split_percentages is not None:
            # Load ALL data - BaseDataset will apply percentage split after this
            total_size = sum(self.__class__.DATASET_SIZE.values())  # 100 + 20 + 30 = 150
            self.annotations = [{'idx': i} for i in range(total_size)]
        else:
            # Load only the specific split's data (DATASET_SIZE is normalized to int by BaseDataset)
            assert isinstance(self.DATASET_SIZE, int), f"DATASET_SIZE should be normalized to int, got {type(self.DATASET_SIZE)}"
            self.annotations = [{'idx': i} for i in range(self.DATASET_SIZE)]

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        return {'input': torch.tensor([idx])}, {'label': torch.tensor([idx])}, {'idx': idx}

    @staticmethod
    def display_datapoint(datapoint, class_labels=None, **kwargs):
        """Mock display method for testing."""
        return f"<div>Mock display for datapoint {datapoint.get('meta_info', {}).get('idx', 'unknown')}</div>"


@pytest.fixture
def mock_dataset_class():
    """Fixture that provides the MockDataset class for testing."""
    return MockDataset


class MockDatasetWithoutPredefinedSplits(BaseDataset):
    """Mock dataset without predefined splits for testing split_percentages functionality."""

    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = 150  # Total size for percentage-based splitting
    INPUT_NAMES = ['input']
    LABEL_NAMES = ['label']
    SHA1SUM = None

    def _init_annotations(self) -> None:
        # Always load everything - splitting is handled by BaseDataset
        self.annotations = [{'idx': i} for i in range(150)]

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        return {'input': torch.tensor([idx])}, {'label': torch.tensor([idx])}, {'idx': idx}

    @staticmethod
    def display_datapoint(datapoint, class_labels=None, **kwargs):
        """Mock display method for testing."""
        return f"<div>Mock display for datapoint {datapoint.get('meta_info', {}).get('idx', 'unknown')}</div>"


@pytest.fixture
def mock_dataset_class_without_predefined_splits():
    """Fixture that provides the MockDatasetWithoutPredefinedSplits class for testing."""
    return MockDatasetWithoutPredefinedSplits
