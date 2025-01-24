import torch
import torchvision
import data
import criteria
import metrics


transforms_config = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                torchvision.transforms.Resize(size=(256, 256), antialias=True),
                ('inputs', 'img_1'),
            ),
            (
                torchvision.transforms.Resize(size=(256, 256), antialias=True),
                ('inputs', 'img_2'),
            ),
            (
                data.transforms.resize.ResizeMaps(size=(256, 256), antialias=True),
                ('labels', 'change_map'),
            ),
        ],
    },
}

collate_fn_config = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {
            'meta_info': {
                'date_1': list,
                'date_2': list,
            },
        },
    },
}

config = {
    'train_dataset': {
        'class': data.datasets.OSCDDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/OSCD",
            'split': "train",
            'transforms_cfg': transforms_config,
            'bands': None,
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 4,
            'num_workers': 4,
            'collate_fn': collate_fn_config,
        },
    },
    'val_dataset': {
        'class': data.datasets.OSCDDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/OSCD",
            'split': "test",
            'transforms_cfg': transforms_config,
            'bands': None,
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'collate_fn': collate_fn_config,
        },
    },
    'test_dataset': None,
    'test_dataloader': None,
    'criterion': {
        'class': criteria.vision_2d.SemanticSegmentationCriterion,
        'args': {
            'class_weights': data.datasets.OSCDDataset.NUM_CLASSES*(1/torch.tensor(data.datasets.OSCDDataset.CLASS_DIST))/torch.sum(1/torch.tensor(data.datasets.OSCDDataset.CLASS_DIST)),
        },
    },
    'metric': {
        'class': metrics.vision_2d.SemanticSegmentationMetric,
        'args': {
            'num_classes': 2,
        },
    },
}
