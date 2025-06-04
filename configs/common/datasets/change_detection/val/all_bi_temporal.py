import torch
import data
import metrics
from configs.common.datasets.change_detection.val.air_change_data_cfg import config as air_change_cfg
from configs.common.datasets.change_detection.val.cdd_data_cfg import config as cdd_cfg
from configs.common.datasets.change_detection.val.levir_cd_data_cfg import config as levir_cd_cfg
from configs.common.datasets.change_detection.val.oscd_data_cfg import config as oscd_cfg
from configs.common.datasets.change_detection.val.sysu_cd_data_cfg import config as sysu_cd_cfg


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
    'metric': {
        'class': metrics.vision_2d.SemanticSegmentationMetric,
        'args': {
            'num_classes': 2,
        },
    },
}
