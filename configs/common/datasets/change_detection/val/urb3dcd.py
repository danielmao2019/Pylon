import torch
import data
import metrics
from ._transforms_cfg import transforms_cfg


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
        'class': metrics.vision_3d.PointCloudConfusionMatrix,
        'args': {
            'num_classes': 7,  # Urb3DCD dataset has 7 classes
        },
    },
}
