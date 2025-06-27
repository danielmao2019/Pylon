import torch
import data
from metrics.wrappers import HybridMetric
from metrics.vision_3d.point_cloud_registration import IsotropicTransformError
from metrics.vision_3d.point_cloud_registration.transform_inlier_ratio import TransformInlierRatio


data_cfg = {
    'eval_dataset': {
        'class': data.datasets.SynthPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'cache_dirname': 'synth_pcr_cache',
            'split': 'val',
            'voxel_size': 10.0,
            'min_points': 512,
            'max_points': 8192,
            'overlap': 0.4,
            'device': 'cpu',
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        (
                            {
                                'class': data.transforms.vision_3d.RandomRigidTransform,
                                'args': {'rot_mag': 45.0, 'trans_mag': 0.5},
                            },
                            [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')],
                        ),
                    ],
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
