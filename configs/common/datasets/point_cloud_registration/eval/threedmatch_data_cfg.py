import torch
import data
from metrics.wrappers import HybridMetric
from metrics.vision_3d.point_cloud_registration import IsotropicTransformError
from metrics.vision_3d.point_cloud_registration.transform_inlier_ratio import TransformInlierRatio


data_cfg = {
    'eval_dataset': {
        'class': data.datasets.ThreeDMatchDataset,
        'args': {
            'data_root': './data/datasets/soft_links/threedmatch',
            'split': 'test',
            'matching_radius': 0.1,
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [],
                },
            },
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
        'class': HybridMetric,
        'args': {
            'metrics_cfg': [
                {
                    'class': IsotropicTransformError,
                    'args': {
                        'use_buffer': False,
                    },
                },
                {
                    'class': TransformInlierRatio,
                    'args': {
                        'threshold': 0.3,
                        'use_buffer': False,
                    },
                },
            ],
        },
    },
}
