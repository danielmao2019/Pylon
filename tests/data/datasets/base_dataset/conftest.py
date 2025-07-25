"""Shared fixtures and helper functions for BaseDataset tests."""

import pytest
from typing import Dict, Any, Tuple
import torch
from data.datasets.base_dataset import BaseDataset


class MockDataset(BaseDataset):
    """Mock dataset for testing BaseDataset functionality."""
    
    SPLIT_OPTIONS = ['train', 'val', 'test']
    INPUT_NAMES = ['input']
    LABEL_NAMES = ['label']
    SHA1SUM = None
    
    def __init__(self, split=None, **kwargs):
        # Set DATASET_SIZE based on split type to match BaseDataset design
        if isinstance(split, tuple):
            # For tuple splits (percentages), use int DATASET_SIZE (total size)
            self.DATASET_SIZE = 150  # Total of train:100 + val:20 + test:30
        else:
            # For string splits, use dict DATASET_SIZE (per-split sizes)
            self.DATASET_SIZE = {'train': 100, 'val': 20, 'test': 30}
        
        super().__init__(split=split, **kwargs)
    
    def _init_annotations(self) -> None:
        # For string splits, use specific split size
        if hasattr(self, 'split') and self.split is not None:
            assert isinstance(self.DATASET_SIZE, dict), "String splits require dict DATASET_SIZE"
            dataset_size = self.DATASET_SIZE[self.split]
        elif hasattr(self, 'split_percentages'):
            assert isinstance(self.DATASET_SIZE, int), "Tuple splits require int DATASET_SIZE"
            dataset_size = self.DATASET_SIZE
        else:
            # This should never happen
            assert False, f"MockDataset must have either 'split' or 'split_percentages' attribute"
        
        self.annotations = [{'idx': i} for i in range(dataset_size)]
    
    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        return {'input': torch.tensor([idx])}, {'label': torch.tensor([idx])}, {'idx': idx}


@pytest.fixture
def mock_dataset_class():
    """Fixture that provides the MockDataset class for testing."""
    return MockDataset
