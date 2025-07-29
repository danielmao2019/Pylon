"""Simple trainer config for eval_viewer integration testing."""
from typing import Dict
import torch
from runners import SupervisedSingleTaskTrainer
from criteria.wrappers import PyTorchCriterionWrapper
from metrics.wrappers import PyTorchMetricWrapper
from optimizers import SingleTaskOptimizer
from schedulers.lr_lambdas import ConstantLambda
from data.datasets.random_datasets import BaseRandomDataset
from data.transforms import Compose


class SimpleTestModel(torch.nn.Module):
    """Simple linear model for testing."""

    def __init__(self) -> None:
        super(SimpleTestModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=2, out_features=1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(inputs, dict), f"{type(inputs)=}"
        assert inputs.keys() == {'x'}, f"{inputs.keys()=}"
        return self.linear(inputs['x'])


# Simple dataset config for testing
dataset_cfg = {
    'class': BaseRandomDataset,
    'args': {
        'num_examples': 16,  # Small dataset for fast execution
        'initial_seed': 42,
        'device': torch.device('cuda'),  # Use CUDA to match framework expectations
        'gen_func_config': {
            'inputs': {
                'x': (
                    torch.rand,
                    {'size': (2,), 'dtype': torch.float32},
                ),
            },
            'labels': {
                'y': (
                    torch.rand,
                    {'size': (1,), 'dtype': torch.float32},
                ),
            },
        },
        'transforms_cfg': None,  # No transforms for simplicity
    },
}

config = {
    'runner': SupervisedSingleTaskTrainer,
    'work_dir': "./logs/tests/runners/eval_viewer/trainer_integration_test",
    'init_seed': 42,
    'epochs': 3,  # Small number of epochs for fast execution
    'train_seeds': [42, 43, 44],
    'val_seeds': [42, 43, 44],
    'test_seed': 42,
    # ==================================================
    # dataloaders
    # ==================================================
    'train_dataset': dataset_cfg,
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 8,
            'num_workers': 0,  # No multiprocessing for testing
        },
    },
    'val_dataset': dataset_cfg,
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,  # Use batch_size=1 for validation (as per CLAUDE.md)
            'num_workers': 0,  # No multiprocessing for testing
        },
    },
    'test_dataset': None,
    'test_dataloader': None,
    # ==================================================
    # model
    # ==================================================
    'model': {
        'class': SimpleTestModel,
        'args': {},
    },
    'criterion': {
        'class': PyTorchCriterionWrapper,
        'args': {
            'criterion': torch.nn.MSELoss(reduction='mean'),
        },
    },
    'metric': {
        'class': PyTorchMetricWrapper,
        'args': {
            'metric': torch.nn.MSELoss(reduction='mean'),
        },
    },
    # ==================================================
    # optimizer
    # ==================================================
    'optimizer': {
        'class': SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': torch.optim.SGD,
                'args': {
                    'lr': 1.0e-02,  # Small learning rate
                },
            },
        },
    },
    'scheduler': {
        'class': torch.optim.lr_scheduler.LambdaLR,
        'args': {
            'lr_lambda': {
                'class': ConstantLambda,
                'args': {},
            },
        },
    },
    # No early stopping for predictable test results
    'early_stopping': None,
}