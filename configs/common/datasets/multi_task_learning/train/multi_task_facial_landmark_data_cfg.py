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
                    'args': {'size': (224, 224), 'antialias': True},
                },
                ('inputs', 'image'),
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
        'class': data.datasets.MultiTaskFacialLandmarkDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/MTFL",
            'split': "train",
            'indices': None,
            'transforms_cfg': transforms_cfg,
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 32,
            'num_workers': 8,
            'collate_fn': collate_fn_cfg
        },
    },
    'test_dataset': {
        'class': data.datasets.MultiTaskFacialLandmarkDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/MTFL",
            'split': "test",
            'indices': None,
            'transforms_cfg': transforms_cfg,
        },
    },
    'test_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 8,
            'collate_fn': collate_fn_cfg,
        },
    },
}