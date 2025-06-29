import torch
import data
import metrics
from configs.common.datasets.change_detection.val._transforms_cfg import transforms_cfg


collate_fn_cfg = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {
            'meta_info': {
                'image_size': torch.Tensor,
                'crop_loc': torch.Tensor,
                'crop_size': torch.Tensor,
            },
        },
    },
}

data_cfg = {
    'val_dataset': {
        'class': data.datasets.AirChangeDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/AirChange",
            'split': "test",
            'transforms_cfg': transforms_cfg(first="RandomCrop", size=(112, 112), resize=(224, 224)),
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
