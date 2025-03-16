from typing import Any, Dict, List, Tuple
import pytest
import torch
from data.datasets.base_dataset import BaseDataset


@pytest.fixture
def SampleDataset():
    """A minimal dataset implementation for testing BaseDataset functionality."""
    class _SampleDataset(BaseDataset):
        SPLIT_OPTIONS = ['train', 'val', 'test', 'weird']
        INPUT_NAMES = ['input']
        LABEL_NAMES = ['label']

        def _init_annotations(self) -> None:
            # all splits are the same
            self.annotations = list(range(100))

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
        ]:
            return {'input': self.annotations[idx]}, {'label': self.annotations[idx]}, {}
    
    return _SampleDataset
