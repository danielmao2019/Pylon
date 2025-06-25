"""Configuration for model3_run_0."""

from data.datasets import COCOStuff164KDataset
import torch

config = {
    'model': {'class': 'Model3', 'args': {}},
    'val_dataset': {
        'class': COCOStuff164KDataset,
        'args': {
            'data_root': './data/datasets/soft_links/COCOStuff164K',
            'split': 'val2017',
            'semantic_granularity': 'coarse',
        }
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {'batch_size': 8}
    },
    'metric': {'class': 'TestMetric', 'args': {}},
    'epochs': 4,
    'seed': 42
}
