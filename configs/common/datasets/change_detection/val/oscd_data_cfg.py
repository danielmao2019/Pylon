import torch
import data
import metrics
from configs.common.datasets.change_detection.val._transforms_cfg import transforms_cfg


collate_fn_cfg = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {
            'meta_info': {
                'date_1': list,
                'date_2': list,
            },
        },
    },
}

data_cfg = {
    'val_dataset': {
        'class': data.datasets.OSCDDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/OSCD",
            'split': "test",
            'transforms_cfg': transforms_cfg(size=(224, 224)),
            'bands': None,
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
    'metric': {
        'class': metrics.vision_2d.SemanticSegmentationMetric,
        'args': {
            'num_classes': 2,
        },
    },
}
