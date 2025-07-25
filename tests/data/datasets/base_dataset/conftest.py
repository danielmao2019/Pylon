"""Shared fixtures and helper functions for BaseDataset tests."""

import pytest
from typing import Dict, Any, Tuple
import torch
from data.datasets.base_dataset import BaseDataset


class MockDataset(BaseDataset):
    """Mock dataset for testing BaseDataset functionality."""
    
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': 100, 'val': 20, 'test': 30}
    INPUT_NAMES = ['input']
    LABEL_NAMES = ['label']
    SHA1SUM = None
    
    def _init_annotations(self) -> None:
        self.annotations = [{'idx': i} for i in range(self.DATASET_SIZE[self.split])]
    
    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        return {'input': torch.tensor([idx])}, {'label': torch.tensor([idx])}, {'idx': idx}


@pytest.fixture
def mock_dataset_class():
    """Fixture that provides the MockDataset class for testing."""
    return MockDataset
