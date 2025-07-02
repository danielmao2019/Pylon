import pytest
from typing import Dict
import tempfile
import shutil
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric


class SimpleMetric(SingleTaskMetric):
    """A simple metric implementation for testing."""
    
    DIRECTION = -1  # Lower is better for MSE (loss metric)

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute MSE score."""
        score = torch.mean((y_pred - y_true) ** 2)
        return {"mse": score}


class SimpleDataset(torch.utils.data.Dataset):
    """A simple dataset for testing."""

    def __init__(self, size=100, device='cuda'):
        self.device = device
        self.data = torch.randn(size, 10, device=device)
        self.labels = torch.randn(size, 1, device=device)
        self.base_seed = 0  # Add base_seed attribute

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'inputs': self.data[idx],
            'labels': self.labels[idx],
            'meta_info': {'idx': idx}
        }

    def set_base_seed(self, seed: int) -> None:
        """Set the base seed for deterministic behavior."""
        self.base_seed = seed


class SimpleModel(torch.nn.Module):
    """A simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def test_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)




@pytest.fixture
def dataloader(dataset):
    """Create a dataloader for testing."""
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


@pytest.fixture
def trainer_cfg(dataloader, device):
    """Create a configuration for testing."""
    return {
        'model': {
            'class': SimpleModel,
            'args': {}
        },
        'train_dataset': None,
        'train_dataloader': None,
        'criterion': None,
        'val_dataset': {
            'class': SimpleDataset,
            'args': {'size': 100, 'device': device}
        },
        'val_dataloader': {
            'class': torch.utils.data.DataLoader,
            'args': {
                'batch_size': 1,
                'shuffle': False
            }
        },
        'metric': {
            'class': SimpleMetric,
            'args': {}
        },
        'optimizer': None,
        'scheduler': None,
        'epochs': 1,
        'init_seed': 42,
        'train_seeds': [42],
        'val_seeds': [42],
        'test_seed': 42
    }


@pytest.fixture
def evaluator_cfg(dataloader, device):
    """Create a configuration for evaluation testing."""
    return {
        'model': {
            'class': SimpleModel,
            'args': {}
        },
        'eval_dataset': {
            'class': SimpleDataset,
            'args': {'size': 100, 'device': device}
        },
        'eval_dataloader': {
            'class': torch.utils.data.DataLoader,
            'args': {
                'batch_size': 1,  # Pylon evaluators expect batch_size=1
                'shuffle': False
            }
        },
        'metric': {
            'class': SimpleMetric,
            'args': {}
        },
        'seed': 42
    }
