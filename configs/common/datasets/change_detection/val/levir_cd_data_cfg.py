import torch
import data
import metrics
from configs.common.datasets.change_detection.val._transforms_cfg import transforms_cfg


collate_fn_cfg = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {
            'meta_info': {
                'image_resolution': torch.Tensor,
            },
        },
    },
}

config = {
    'val_dataset': {
        'class': data.datasets.LevirCdDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/LEVIR-CD",
            'split': "val",
            'transforms_cfg': transforms_cfg(size=(224, 224)),
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'collate_fn': collate_fn_cfg,
        },
    },
    'test_dataset': {
        'class': data.datasets.LevirCdDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/LEVIR-CD",
            'split': "test",
            'transforms_cfg': transforms_cfg,
        },
    },
    'test_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'collate_fn': collate_fn_cfg,
        },
    },
    'metric': {
        'class': metrics.vision_2d.SemanticSegmentationMetric,
        'args': {
            'num_classes': 2,
        },
    },
}
