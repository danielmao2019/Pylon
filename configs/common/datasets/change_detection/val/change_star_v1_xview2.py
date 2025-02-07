import data.collators.change_star_collator
import torch
import data
import metrics


transforms_config = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                data.transforms.resize.ResizeMaps(size=(256, 256), antialias=True),
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'lbl_1'), ('labels', 'lbl_2')]
            ),
        ],
    },
}

collate_fn_cfg = {
    'class': data.collators.ChangeStarCollator,
    'args': {
        'method': "eval",
    },
}

config = {
    'val_dataset': {
        'class': data.datasets.xView2Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/xView2",
            'split': "test",
            'transforms_cfg': transforms_config,
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
        'class': data.datasets.xView2Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/xView2",
            'split': "hold",
            'transforms_cfg': transforms_config,
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
        'class': metrics.vision_2d.ChangeStarMetric,
        'args': {},
    },
}
