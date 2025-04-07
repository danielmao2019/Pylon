import os
import sys
import pytest
import torch
import tempfile
import shutil
import json
from typing import Dict, Any, List
from utils.builders import build_from_config
from utils.io import save_json

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics.base_metric import BaseMetric
from utils.logging.logger import Logger


class SimpleMetric(BaseMetric):
    """A simple metric implementation for testing."""
    
    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute MSE score."""
        score = torch.mean((y_pred - y_true) ** 2)
        return {"mse": score}
    
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Update the metric buffer with the current batch's score."""
        score = self._compute_score(y_pred, y_true)
        # Detach and move to CPU
        score = {k: v.detach().cpu() for k, v in score.items()}
        # Add to buffer
        self.add_to_buffer(score)
        return score
    
    def summarize(self, output_path=None) -> Dict[str, float]:
        """Calculate average score from buffer and optionally save to file."""
        if not self.buffer:
            return {"mse": 0.0}
        
        # Calculate average score
        mse_scores = [score["mse"] for score in self.buffer]
        avg_score = sum(mse_scores) / len(mse_scores)
        result = {"mse": avg_score}
        
        # Save to file if path is provided
        if output_path:
            save_json(obj=result, filepath=output_path)
        
        return result


class SimpleDataset(torch.utils.data.Dataset):
    """A simple dataset for testing."""
    
    def __init__(self, size=100, device='cuda'):
        self.device = device
        self.data = torch.randn(size, 10, device=device)
        self.labels = torch.randn(size, 1, device=device)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'inputs': self.data[idx], 'labels': self.labels[idx]}


class SimpleModel(torch.nn.Module):
    """A simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def test_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)


@pytest.fixture
def device():
    """Get the device to use for testing."""
    return 'cuda'


@pytest.fixture
def dataset(device):
    """Create a simple dataset for testing."""
    return SimpleDataset(device=device)


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
                'batch_size': 32,
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
        'train_seeds': [42]
    }


@pytest.fixture
def evaluator_cfg(dataloader, device):
    """Create a configuration for evaluation testing."""
    return {
        'model': {
            'class': SimpleModel,
            'args': {}
        },
        'eval_dataloader': {
            'class': torch.utils.data.DataLoader,
            'args': {
                'dataset': {
                    'class': SimpleDataset,
                    'args': {'size': 100, 'device': device}
                },
                'batch_size': 32,
                'shuffle': False
            }
        },
        'metric': {
            'class': SimpleMetric,
            'args': {}
        },
        'seed': 42
    }
