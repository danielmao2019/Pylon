from typing import Any, Dict, List, Tuple
import pytest
import torch
from data.datasets.base_dataset import BaseDataset
from data.transforms.compose import Compose
from data.transforms.random_noise import RandomNoise


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
            # Create a random tensor for testing transforms
            # Use the annotation index as the random seed for reproducibility
            torch.manual_seed(self.annotations[idx])
            tensor = torch.randn(3, 32, 32)  # Random noise
            return {'input': tensor}, {'label': self.annotations[idx]}, {}

    return _SampleDataset


@pytest.fixture
def random_transforms():
    """Random transforms for testing cache behavior with randomization."""
    noise = RandomNoise(std=0.2)  # Add significant noise for visible effect

    return {
        'class': Compose,
        'args': {
            'transforms': [
                (noise, [('inputs', 'input')]),
            ]
        }
    }
