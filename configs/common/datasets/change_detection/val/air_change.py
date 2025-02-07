import torch
import data
import metrics


transforms_cfg = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                data.transforms.resize.ResizeMaps(size=(256, 256), antialias=True),
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')]
            ),
        ],
    },
}

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

config = {
    'val_dataset': {
        'class': data.datasets.AirChangeDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/AirChange",
            'split': "test",
            'transforms_cfg': transforms_cfg,
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
