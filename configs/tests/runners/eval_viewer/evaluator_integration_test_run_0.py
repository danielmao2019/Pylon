"""Simple evaluator config for eval_viewer integration testing."""
from typing import Dict
import torch
from runners.evaluators import BaseEvaluator
from metrics.wrappers import PyTorchMetricWrapper
from data.datasets.random_datasets import BaseRandomDataset


class SimpleEvalModel(torch.nn.Module):
    """Simple linear model for evaluation testing."""

    def __init__(self) -> None:
        super(SimpleEvalModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=2, out_features=1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(inputs, dict), f"{type(inputs)=}"
        assert inputs.keys() == {'x'}, f"{inputs.keys()=}"
        return self.linear(inputs['x'])


# Simple dataset config for evaluation testing
eval_dataset_cfg = {
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
    'runner': BaseEvaluator,
    'work_dir': "./logs/tests/runners/eval_viewer/evaluator_integration_test",
    'seed': 42,
    # ==================================================
    # evaluation dataset and dataloader
    # ==================================================
    'eval_dataset': eval_dataset_cfg,
    'eval_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,  # Use batch_size=1 for evaluation (as per CLAUDE.md)
            'num_workers': 0,  # No multiprocessing for testing
        },
    },
    # ==================================================
    # model and metric
    # ==================================================
    'model': {
        'class': SimpleEvalModel,
        'args': {},
    },
    'metric': {
        'class': PyTorchMetricWrapper,
        'args': {
            'metric': torch.nn.MSELoss(reduction='mean'),
        },
    },
}