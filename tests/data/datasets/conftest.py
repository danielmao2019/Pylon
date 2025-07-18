from typing import Any, Dict, Tuple
import pytest
import torch
from data.datasets.base_dataset import BaseDataset
from data.transforms.compose import Compose
from data.transforms.random_noise import RandomNoise


def pytest_addoption(parser):
    """Add custom command line options for dataset testing."""
    parser.addoption(
        "--samples",
        action="store",
        default=None,
        type=int,
        help="Maximum number of datapoints to test per dataset (default: test all or existing limits)"
    )


@pytest.fixture
def max_samples(request):
    """Fixture to get the maximum number of samples to test from command line."""
    return request.config.getoption("--samples")


@pytest.fixture
def get_samples_to_test():
    """
    Fixture that provides helper function to determine how many samples to test.
    """
    def _get_samples_to_test(dataset_length: int, max_samples: int = None) -> int:
        """
        Helper function to determine how many samples to test based on command line args.

        Args:
            dataset_length: Total number of samples in the dataset
            max_samples: Value from --samples command line argument (can be None)

        Returns:
            Number of samples to test (full dataset if --samples not provided)
        """
        if max_samples is not None:
            return min(dataset_length, max_samples)
        else:
            return dataset_length
    return _get_samples_to_test


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
