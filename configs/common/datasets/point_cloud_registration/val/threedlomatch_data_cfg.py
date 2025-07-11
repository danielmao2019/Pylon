import torch
import data
import utils


data_cfg = {
    'val_dataset': {
        'class': data.datasets.ThreeDLoMatchDataset,
        'args': {
            'data_root': './data/datasets/soft_links/threedmatch',
            'split': 'val',
            'matching_radius': 0.1,
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        (
                            {
                                'class': utils.point_cloud_ops.random_select.RandomSelect,
                                'args': {
                                    'count': 5000,
                                },
                            },
                            [('inputs', 'src_pc')],
                        ),
                        (
                            {
                                'class': utils.point_cloud_ops.random_select.RandomSelect,
                                'args': {
                                    'count': 5000,
                                },
                            },
                            [('inputs', 'tgt_pc')],
                        ),
                    ],
                },
            },
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'shuffle': False,
        },
    },
}