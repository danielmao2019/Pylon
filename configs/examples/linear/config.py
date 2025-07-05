from typing import Dict
import torch
from runners import SupervisedSingleTaskTrainer
from criteria.wrappers import PyTorchCriterionWrapper
from metrics.wrappers import PyTorchMetricWrapper
from optimizers import SingleTaskOptimizer
from schedulers.lr_lambdas import ConstantLambda

from configs.examples.linear.dataset_cfg import dataset_cfg


class TestModel(torch.nn.Module):

    def __init__(self) -> None:
        super(TestModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=2, out_features=2)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(inputs, dict), f"{type(inputs)=}"
        assert inputs.keys() == {'x'}, f"{inputs.keys()=}"
        return self.linear(inputs['x'])


config = {
    'runner': SupervisedSingleTaskTrainer,
    'work_dir': "./logs/examples/linear",
    'init_seed': 0,
    'epochs': 10,
    'train_seeds': [0] * 10,
    'val_seeds': [0] * 10,
    'test_seed': 0,
    # ==================================================
    # dataloaders
    # ==================================================
    'train_dataset': dataset_cfg,
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 4,
            'num_workers': 8,
        },
    },
    'val_dataset': dataset_cfg,
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'num_workers': 8,
        },
    },
    'test_dataset': None,
    'test_dataloader': None,
    # ==================================================
    # model
    # ==================================================
    'model': {
        'class': TestModel,
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
                    'lr': 1.0e-01,
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
}
