import torch
import data
import utils


data_cfg = {
    'train_dataset': {
        'class': data.datasets.ThreeDLoMatchDataset,
        'args': {
            'data_root': './data/datasets/soft_links/threedmatch',
            'split': 'train',
            'matching_radius': 0.1,
            'overlap_threshold': 0.3,
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
                        (
                            {
                                'class': data.transforms.vision_3d.RandomRigidTransform,
                                'args': {
                                    'rot_mag': 45.0,
                                    'trans_mag': 0.5,
                                    'method': 'Rodrigues',
                                },
                            },
                            [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')],
                        ),
                        (
                            {
                                'class': data.transforms.vision_3d.GaussianPosNoise,
                                'args': {
                                    'std': 0.01,
                                },
                            },
                            [('inputs', 'src_pc')],
                        ),
                        (
                            {
                                'class': data.transforms.vision_3d.GaussianPosNoise,
                                'args': {
                                    'std': 0.01,
                                },
                            },
                            [('inputs', 'tgt_pc')],
                        ),
                    ],
                },
            },
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'shuffle': True,
        },
    },
    'criterion': None,
}