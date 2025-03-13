import torch
import data
import criteria


config = {
    'train_dataset': {
        'class': data.datasets.SLPCCDDataset,
        'args': {
            'data_root': "./data/datasets/SLPCCD",
            'split': "train",
            'num_points': 8192,
            'random_subsample': True,
            'use_hierarchy': True,
            'hierarchy_levels': 3,
            'knn_size': 16,
            'cross_knn_size': 16,
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 4,
            'num_workers': 4,
            'shuffle': True,
            'pin_memory': True,
        },
    },
    'criterion': {
        'class': criteria.vision_3d.PointCloudSegmentationCriterion,
        'args': {
            'ignore_value': -1,  # SLPCCD dataset uses -1 as ignore value
            'class_weights': None,  # Can be adjusted based on class distribution
        },
    },
}
