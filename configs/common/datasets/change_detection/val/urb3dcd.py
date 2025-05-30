import torch
import data
import metrics


config = {
    'val_dataset': {
        'class': data.datasets.Urb3DCDDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/Urb3DCD",
            'split': "val",
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
        },
    },
    'metric': {
        'class': metrics.common.ConfusionMatrix,
        'args': {
            'num_classes': 7,  # Urb3DCD dataset has 7 classes
        },
    },
}
