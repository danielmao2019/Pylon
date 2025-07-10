import torch
import data


data_cfg = {
    'test_dataset': {
        'class': data.datasets.ThreeDMatchDataset,
        'args': {
            'data_root': './data/datasets/soft_links/threedmatch',
            'split': 'test',
            'num_points': 5000,
            'matching_radius': 0.1,
            'overlap_threshold': 0.3,
            'benchmark_mode': '3DMatch',
            # No transforms for evaluation
        },
    },
    'test_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'shuffle': False,
        },
    },
}
