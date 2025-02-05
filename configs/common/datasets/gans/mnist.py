import criteria.wrappers.pytorch_criterion_wrapper
import torch
import data
import criteria
import metrics


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
        'class': data.datasets.GANDataset,
        'args': {
            'source': data.datasets.MNISTDataset(
                data_root="./data/datasets/soft_links/MNIST",
                split="train",
            ),
            'latent_dim': 128,
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
        'class': data.datasets.MNISTDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/MNIST",
            'split': "test",
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
        'class': data.datasets.MNISTDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/MNIST",
            'split': "test",
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
        'class': criteria.wrappers.PyTorchCriterionWrapper,
        'args': {
            'criterion': torch.nn.BCELoss(),
        },
    },
    'metric': {
        'class': metrics.wrappers.MultiTaskMetric,
        'args': {
            'metric_configs': {
                task: metrics.common.ConfusionMatrix(num_classes=2)
                for task in data.datasets.MNISTDataset.LABEL_NAMES[1:]
            },
        },
    },
}
