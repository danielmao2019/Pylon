from typing import Any, Dict, List, Tuple
import pytest
import torch
from data.datasets.base_dataset import BaseDataset
from data.transforms.compose import Compose
from data.transforms.random_rotation import RandomRotation
from data.transforms.flip import Flip


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
            # Create a simple tensor for testing transforms
            tensor = torch.ones(3, 32, 32) * self.annotations[idx]
            return {'input': tensor}, {'label': self.annotations[idx]}, {}
    
    return _SampleDataset


@pytest.fixture
def random_transforms():
    """Random transforms for testing cache behavior with randomization."""
    rotation = RandomRotation(range=(-30, 30))  # Random rotation between -30 and 30 degrees
    flip = Flip(axis=2)  # Horizontal flip
    
    return Compose([
        (rotation, ('inputs', 'input')),
        (flip, ('inputs', 'input')),
    ])
