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
            'dataset_size': 500,  # Smaller for evaluation
            'overlap_range': (0.0, 1.0),  # GeoTransformer doesn't use specific overlap ranges
            'matching_radius': 0.05,  # Radius for correspondence finding
            'rotation_mag': 45.0,  # GeoTransformer synthetic transform parameters
            'translation_mag': 0.5,  # GeoTransformer synthetic transform parameters
            'device': 'cpu',
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                            # clamp to min max points 512 - 8192
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
