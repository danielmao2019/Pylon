import torch
import data
import metrics


config = {
    'val_dataset': {
        'class': data.datasets.SLPCCDDataset,
        'args': {
            'data_root': "./data/datasets/SLPCCD",
            'split': "val",
            'num_points': 8192,
            'random_subsample': True,
            'use_hierarchy': True,
            'hierarchy_levels': 3,
            'knn_size': 16,
            'cross_knn_size': 16,
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'shuffle': False,
            'pin_memory': True,
        },
    },
    'test_dataset': {
        'class': data.datasets.SLPCCDDataset,
        'args': {
            'data_root': "./data/datasets/SLPCCD",
            'split': "test",
            'num_points': 8192,
            'random_subsample': True,
            'use_hierarchy': True,
            'hierarchy_levels': 3,
            'knn_size': 16,
            'cross_knn_size': 16,
        },
    },
    'test_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,  # Using batch size 1 for testing to avoid padding issues
            'num_workers': 4,
            'shuffle': False,
            'pin_memory': True,
        },
    },
    'metric': {
        'class': metrics.common.ConfusionMatrix,
        'args': {
            'num_classes': 2,  # SLPCCD dataset has 2 classes (unchanged/changed)
        },
    },
}
