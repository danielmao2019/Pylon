import torch
import data
import criteria
from ._transforms_cfg import transforms_cfg


config = {
    'train_dataset': {
        'class': data.datasets.Urb3DCDDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/Urb3DCD",
            'split': "train",
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 4,
            'num_workers': 4,
        },
    },
    'criterion': {
        'class': criteria.vision_2d.SemanticSegmentationCriterion,
        'args': {},
    },
}
