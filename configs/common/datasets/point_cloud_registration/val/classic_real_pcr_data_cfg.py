import torch
from data.datasets import SynthPCRDataset
from metrics.vision_3d import RegistrationRecall


data_cfg = {
    'eval_dataset': {
        'class': RealPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'split': 'val',
            'voxel_size': 10.0,
            'min_points': 256,
            'max_points': 8192,
            'device': 'cpu',
        },
    },
    'eval_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
        },
    },
    'metric': {
        'class': RegistrationRecall,
        'args': {
            'rot_threshold_deg': 5.0,
            'trans_threshold_m': 0.3,
        },
    },
}
