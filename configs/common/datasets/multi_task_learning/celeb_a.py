import criteria.wrappers.pytorch_criterion_wrapper
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
                torchvision.transforms.Resize(size=(64, 64), antialias=True),
                ('inputs', 'image'),
            ),
        ],
    },
}

collate_fn_config = {
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
            'transforms_cfg': transforms_config,
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 256,
            'num_workers': 0,
            'collate_fn': collate_fn_config,
        },
    },
    'val_dataset': {
        'class': data.datasets.CelebADataset,
        'args': {
            'data_root': "./data/datasets/soft_links/celeb-a",
            'split': "val",
            'indices': None,
            'transforms_cfg': transforms_config,
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 256,
            'num_workers': 0,
            'collate_fn': collate_fn_config,
        },
    },
    'test_dataset': {
        'class': data.datasets.CelebADataset,
        'args': {
            'data_root': "./data/datasets/soft_links/celeb-a",
            'split': "test",
            'indices': None,
            'transforms_cfg': transforms_config,
        },
    },
    'test_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 256,
            'num_workers': 0,
            'collate_fn': collate_fn_config,
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
