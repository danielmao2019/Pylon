import torch
import data
from metrics.vision_3d.point_cloud_registration.transform_inlier_ratio import TransformInlierRatio
from metrics.vision_3d.point_cloud_registration.isotropic_transform_error import IsotropicTransformError
from metrics.wrappers.hybrid_metric import HybridMetric


data_cfg = {
    'eval_dataset': {
        'class': data.datasets.KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'val',
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
