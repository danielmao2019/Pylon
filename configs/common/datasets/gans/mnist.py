import criteria.wrappers.pytorch_criterion_wrapper
import torch
import data
import criteria


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
    'val_dataset': None,
    'val_dataloader': None,
    'test_dataset': None,
    'test_dataloader': None,
    'criterion': {
        'class': criteria.wrappers.PyTorchCriterionWrapper,
        'args': {
            'criterion': torch.nn.BCELoss(),
        },
    },
    'metric': None,
}
