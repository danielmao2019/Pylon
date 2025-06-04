import data.collators.change_star_collator
import torch
import data
import metrics
from configs.common.datasets.change_detection.val.air_change_data_cfg import config as air_change_cfg
from configs.common.datasets.change_detection.val.cdd_data_cfg import config as cdd_cfg
from configs.common.datasets.change_detection.val.levir_cd_data_cfg import config as levir_cd_cfg
from configs.common.datasets.change_detection.val.oscd_data_cfg import config as oscd_cfg
from configs.common.datasets.change_detection.val.sysu_cd_data_cfg import config as sysu_cd_cfg


transforms_cfg = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                data.transforms.vision_2d.RandomCrop(size=(224, 224)),
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'lbl_1'), ('labels', 'lbl_2')]
            ),
        ],
    },
}

change_star_v1_collate_fn_cfg = {
    'class': data.collators.ChangeStarCollator,
    'args': {
        'method': "eval",
    },
}

data_cfg = {
    'val_datasets': [{
        'class': data.datasets.xView2Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/xView2",
            'split': "test",
            'transforms_cfg': transforms_cfg,
        },
    }] + list(map(lambda x: x['val_dataset'], [air_change_cfg, cdd_cfg, levir_cd_cfg, oscd_cfg, sysu_cd_cfg])),
    'val_dataloaders': [{
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'collate_fn': change_star_v1_collate_fn_cfg,
        },
    }] + list(map(lambda x: {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'collate_fn': x['val_dataloader']['args']['collate_fn'],
        },
    }, [air_change_cfg, cdd_cfg, levir_cd_cfg, oscd_cfg, sysu_cd_cfg])),
    'metric': {
        'class': metrics.vision_2d.ChangeStarMetric,
        'args': {},
    },
}
