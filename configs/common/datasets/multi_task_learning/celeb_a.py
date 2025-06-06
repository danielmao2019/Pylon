import criteria.wrappers.pytorch_criterion_wrapper
import torch
import torchvision
import data
import criteria
import metrics


transforms_cfg = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                {
                    'class': torchvision.transforms.Resize,
                    'args': {'size': (64, 64), 'antialias': True},
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

config = {
    'train_dataset': {
        'class': data.datasets.CelebADataset,
        'args': {
            'data_root': "./data/datasets/soft_links/celeb-a",
            'split': "train",
            'indices': None,
            'transforms_cfg': transforms_cfg,
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 256,
            'num_workers': 0,
            'collate_fn': collate_fn_cfg,
        },
    },
    'val_dataset': {
        'class': data.datasets.CelebADataset,
        'args': {
            'data_root': "./data/datasets/soft_links/celeb-a",
            'split': "val",
            'indices': None,
            'transforms_cfg': transforms_cfg,
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 256,
            'num_workers': 0,
            'collate_fn': collate_fn_cfg,
        },
    },
    'test_dataset': {
        'class': data.datasets.CelebADataset,
        'args': {
            'data_root': "./data/datasets/soft_links/celeb-a",
            'split': "test",
            'indices': None,
            'transforms_cfg': transforms_cfg,
        },
    },
    'test_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 256,
            'num_workers': 0,
            'collate_fn': collate_fn_cfg,
        },
    },
    'criterion': {
        'class': criteria.wrappers.MultiTaskCriterion,
        'args': {
            'criterion_configs': {
                task: criteria.wrappers.PyTorchCriterionWrapper(criterion=torch.nn.CrossEntropyLoss())
                for task in data.datasets.CelebADataset.LABEL_NAMES[1:]
            },
        },
    },
    'metric': {
        'class': metrics.wrappers.MultiTaskMetric,
        'args': {
            'metric_configs': {
                task: metrics.common.ConfusionMatrix(num_classes=2)
                for task in data.datasets.CelebADataset.LABEL_NAMES[1:]
            },
        },
    },
}
