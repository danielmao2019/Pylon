"""The purpose of this set of test cases is to compare training performance against other frameworks,
including the native PyTorch for loop.
"""
from typing import List, Dict
from .supervised_single_task_trainer import SupervisedSingleTaskTrainer
import os
import json
import torch
import torchvision
import data
import criteria
import metrics
import schedulers


dataset_config = {
    'class': data.datasets.random_datasets.ClassificationRandomDataset,
    'args': {
        'num_examples': 32,
        'num_classes': 10,
        'image_res': (224, 224),
        'initial_seed': 0,
    },
}

class TestModel(torch.nn.Module):

    def __init__(self, model: torch.nn.Module) -> None:
        super(TestModel, self).__init__()
        self.model = model

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert type(inputs) == dict, f"{type(inputs)=}"
        assert len(inputs) == 1, f"{inputs.keys()=}"
        return self.model(list(inputs.values())[0])


config = {
    'work_dir': "./logs/test_supervised_single_task_trainer",
    'init_seed': 0,
    'epochs': 25,
    'train_seeds': [0] * 25,
    # ==================================================
    # dataloaders
    # ==================================================
    'train_dataset': dataset_config,
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 32,
            'num_workers': 8,
        },
    },
    'val_dataset': dataset_config,
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'num_workers': 8,
        },
    },
    'test_dataset': dataset_config,
    'test_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'num_workers': 8,
        },
    },
    # ==================================================
    # model
    # ==================================================
    'model': {
        'class': TestModel,
        'args': {
            'model': torchvision.models.AlexNet(num_classes=10),
        },
    },
    'criterion': {
        'class': criteria.PyTorchCriterionWrapper,
        'args': {
            'criterion': torch.nn.CrossEntropyLoss(reduction='mean'),
        },
    },
    'metric': {
        'class': metrics.ConfusionMatrix,
        'args': {
            'num_classes': 10,
        },
    },
    # ==================================================
    # optimizer
    # ==================================================
    'optimizer': {
        'class': torch.optim.SGD,
        'args': {
            'lr': 1e-03,
        },
    },
    'scheduler': {
        'class': torch.optim.lr_scheduler.LambdaLR,
        'args': {
            'lr_lambda': {
                'class': schedulers.WarmupLambda,
                'args': {},
            },
        },
    },
}

def test_supervised_single_task_trainer() -> None:
    if os.path.isdir(config['work_dir']):
        os.system(' '.join(["rm", "-r", config['work_dir']]))
    trainer = SupervisedSingleTaskTrainer(config=config)
    trainer.train()
    acc: List[float] = []
    for idx in range(config['epochs']):
        epoch_dir = os.path.join(config['work_dir'], f"epoch_{idx}")
        with open(os.path.join(epoch_dir, "validation_scores.json")) as f:
            acc.append(json.load(f)['accuracy'])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(acc)
    plt.savefig(os.path.join(config['work_dir'], "acc.png"))
