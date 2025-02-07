import torch
import data
import metrics
from .air_change import config as air_change_cfg
from .cdd import config as cdd_cfg
from .levir_cd import config as levir_cd_cfg
from .oscd import config as oscd_cfg
from .sysu_cd import config as sysu_cd_cfg


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
    'val_datasets': [
        air_change_cfg['val_dataset'],
        cdd_cfg['val_dataset'],
        levir_cd_cfg['val_dataset'],
        oscd_cfg['val_dataset'],
        sysu_cd_cfg['val_dataset'],
    ],
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
