import pytest
from typing import Dict, List, Any
import tempfile
import shutil
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric
from data.collators.base_collator import BaseCollator


class SimpleMetric(SingleTaskMetric):
    """A simple metric implementation for testing."""
    
    DIRECTIONS = {"mse": -1}  # Lower is better for MSE (loss metric)

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute MSE score."""
        assert isinstance(y_pred, torch.Tensor), f"Expected torch.Tensor, got {type(y_pred)}"
        assert isinstance(y_true, torch.Tensor), f"Expected torch.Tensor, got {type(y_true)}"
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
            'inputs': {'data': self.data[idx]},  # Structure inputs properly
            'labels': {'target': self.labels[idx]},  # Structure labels properly  
            'meta_info': {'idx': idx}  # Keep idx as int
        }

    def set_base_seed(self, seed: int) -> None:
        """Set the base seed for deterministic behavior."""
        self.base_seed = seed


class SimpleModel(torch.nn.Module):
    """A simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, inputs):
        x = inputs['data']
        return {'output': self.linear(x)}


@pytest.fixture
def test_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)


@pytest.fixture
def device():
    """Create device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def dataset(device):
    """Create a simple dataset for testing."""
    return SimpleDataset(size=100, device=device)


@pytest.fixture
def dataloader(dataset):
    """Create a dataloader for testing."""
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=BaseCollator())


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
                'batch_size': 1,  # MUST use batch_size=1 for validation/evaluation
                'shuffle': False,
                'collate_fn': {
                    'class': BaseCollator,
                    'args': {},
                },
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
                'batch_size': 1,  # MUST use batch_size=1 for validation/evaluation
                'shuffle': False,
                'collate_fn': {
                    'class': BaseCollator,
                    'args': {},
                },
            }
        },
        'metric': {
            'class': SimpleMetric,
            'args': {}
        },
        'seed': 42
    }
