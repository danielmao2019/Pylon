import torch
import data


data_cfg = {
    'val_dataset': {
        'class': data.datasets.SynthPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'split': 'val',
            'rot_mag': 45.0,
            'trans_mag': 0.5,
            'voxel_size': 10.0,
            'min_points': 256,
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
        },
    },
    'metric': None,
}
