import data.collators.change_star_collator
import torch
import data
import criteria
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

collate_fn_train_config = {
    'class': data.collators.ChangeStarCollator,
    'args': {
        'method': "train",
    },
}

collate_fn_eval_config = {
    'class': data.collators.ChangeStarCollator,
    'args': {
        'method': "eval",
    },
}

config = {
    'train_dataset': {
        'class': data.datasets.xView2Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/xView2",
            'split': "train",
            'transforms_cfg': transforms_config,
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 4,
            'num_workers': 4,
            'collate_fn': collate_fn_train_config,
        },
    },
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
            'collate_fn': collate_fn_eval_config,
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
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'collate_fn': collate_fn_eval_config,
        },
    },
    'criterion': {
        'class': criteria.vision_2d.ChangeStarCriterion,
        'args': {},
    },
    'metric': {
        'class': metrics.vision_2d.ChangeStarMetric,
        'args': {},
    },
}
