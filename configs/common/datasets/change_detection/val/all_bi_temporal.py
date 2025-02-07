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
    'val_dataloaders': list(map(lambda x: {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'collate_fn': x['val_dataloader']['args']['collate_fn'],
        },
    }, [air_change_cfg, cdd_cfg, levir_cd_cfg, oscd_cfg, sysu_cd_cfg])),
    'metric': list(map(lambda x: x['metric'], [air_change_cfg, cdd_cfg, levir_cd_cfg, oscd_cfg, sysu_cd_cfg])),
}
