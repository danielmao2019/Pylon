import torch
import data
import metrics
from configs.common.datasets.change_detection.val._transforms_cfg import transforms_cfg


collate_fn_cfg = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {},
    },
}

data_cfg = {
    'val_dataset': {
        'class': data.datasets.SYSU_CD_Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/SYSU-CD",
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
        'class': data.datasets.SYSU_CD_Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/SYSU-CD",
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
