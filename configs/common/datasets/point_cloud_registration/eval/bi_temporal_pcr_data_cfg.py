import torch
import data
from metrics.wrappers import HybridMetric
from metrics.vision_3d.point_cloud_registration import IsotropicTransformError
from metrics.vision_3d.point_cloud_registration.transform_inlier_ratio import TransformInlierRatio


data_cfg = {
    'eval_dataset': {
        'class': data.datasets.BiTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'cache_filepath': './data/datasets/soft_links/ivision-pcr-data/../bi_temporal_pcr_cache.json',
            'split': 'val',
            'dataset_size': 500,  # Smaller for evaluation
            'rotation_mag': 45.0,  # GeoTransformer synthetic transform parameters
            'translation_mag': 0.5,  # GeoTransformer synthetic transform parameters
            'matching_radius': 0.05,  # Radius for correspondence finding
            'overlap_range': (0.0, 1.0),  # GeoTransformer doesn't use specific overlap ranges
            'min_points': 512,  # Minimum points filter for cache generation
            'device': 'cpu',
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        {
                            'op': {
                                'class': data.transforms.vision_3d.Clamp,
                                'args': {'max_points': 8192},
                            },
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
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
