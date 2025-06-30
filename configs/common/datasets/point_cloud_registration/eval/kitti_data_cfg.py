import torch
import data
from metrics.vision_3d.point_cloud_registration.inlier_ratio import InlierRatio
from metrics.vision_3d.point_cloud_registration.isotropic_transform_error import IsotropicTransformError
from metrics.wrappers.hybrid_metric import HybridMetric


data_cfg = {
    'eval_dataset': {
        'class': data.datasets.KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'val',
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
                    'args': {},
                },
                {
                    'class': InlierRatio,
                    'args': {
                        'threshold': 0.3,
                    },
                },
            ],
        },
    },
}
