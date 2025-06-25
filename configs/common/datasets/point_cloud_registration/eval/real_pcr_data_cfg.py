import torch
import data
from metrics.wrappers import HybridMetric
from metrics.vision_3d.point_cloud_registration import IsotropicTransformError, InlierRatio


data_cfg = {
    'eval_dataset': {
        'class': data.datasets.RealPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'cache_dirname': 'real_pcr_cache',
            'split': 'val',
            'voxel_size': 10.0,
            'min_points': 512,
            'max_points': 8192,
            'overlap': 0.4,
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        (
                            {
                                'class': data.transforms.vision_3d.PCRTranslation,
                                'args': {},
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
