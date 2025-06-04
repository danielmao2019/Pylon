import torch
import data
import criteria


data_cfg = {
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
            'batch_size': 1,
        },
    },
    'criterion': {
        'class': criteria.vision_3d.PointCloudSegmentationCriterion,
        'args': {
            'ignore_value': -1,  # Urb3DCD dataset uses -1 as ignore value
            'class_weights': None,  # Can be adjusted based on class distribution
        },
    },
}
