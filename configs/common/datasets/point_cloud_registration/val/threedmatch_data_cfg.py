import torch
import data


data_cfg = {
    'val_dataset': {
        'class': data.datasets.ThreeDMatchDataset,
        'args': {
            'data_root': './data/datasets/soft_links/threedmatch',
            'split': 'val',
            'matching_radius': 0.1,
            'overlap_threshold': 0.3,
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        (
                            {
                                'class': data.transforms.vision_3d.RandomSelect,
                                'args': {
                                    'percentage': 0.7,  # Keep 70% of points for 5000 target
                                },
                            },
                            [('inputs', 'src_pc')],
                        ),
                        (
                            {
                                'class': data.transforms.vision_3d.RandomSelect,
                                'args': {
                                    'percentage': 0.7,
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
