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
                ('labels', 'semantic_segmentation'),
            ),
            (
                {
                    'class': data.transforms.vision_2d.ResizeNormals,
                    'args': {'target_size': (512, 512)},
                },
                ('labels', 'normal_estimation'),
            ),
            (
                {
                    'class': data.transforms.vision_2d.ResizeMaps,
                    'args': {'size': (512, 512), 'antialias': True},
                },
                ('labels', 'saliency_estimation'),
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
        'class': data.datasets.PASCALContextDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/PASCAL_Context",
            'split': "train",
            'indices': None,
            'transforms_cfg': transforms_cfg,
            'num_human_parts': 6,
            'area_thres': 0,
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
        'class': data.datasets.PASCALContextDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/PASCAL_Context",
            'split': "val",
            'indices': None,
            'transforms_cfg': transforms_cfg,
            'num_human_parts': 6,
            'area_thres': 0,
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