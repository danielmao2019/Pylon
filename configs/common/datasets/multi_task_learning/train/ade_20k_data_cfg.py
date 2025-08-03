import torch
import torchvision
import data

transforms_cfg = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                {
                    'class': torchvision.transforms.Resize,
                    'args': {'size': (512, 512), 'antialias': True},
                },
                ('inputs', 'image'),
            ),
            (
                {
                    'class': data.transforms.vision_2d.ResizeMaps,
                    'args': {'size': (512, 512), 'interpolation': 'nearest', 'antialias': True},
                },
                ('labels', 'object_cls_mask'),
            ),
            (
                {
                    'class': data.transforms.vision_2d.ResizeMaps,
                    'args': {'size': (512, 512), 'interpolation': 'nearest', 'antialias': True},
                },
                ('labels', 'object_ins_mask'),
            ),
        ],
    },
}

collate_fn_cfg = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {
            'meta_info': {
                'image_resolution': torch.tensor,
            },
        },
    },
}

data_cfg = {
    'train_dataset': {
        'class': data.datasets.ADE20KDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/ADE20K",
            'split': "training",
            'indices': None,
            'transforms_cfg': transforms_cfg,
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 8,
            'num_workers': 8,
            'collate_fn': collate_fn_cfg
        },
    },
    'val_dataset': {
        'class': data.datasets.ADE20KDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/ADE20K",
            'split': "validation",
            'indices': None,
            'transforms_cfg': transforms_cfg,
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 8,
            'collate_fn': collate_fn_cfg,
        },
    },
}