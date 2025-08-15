"""
GMCNet training dataset configuration for ModelNet40.

This config replicates the original GMCNet training setup:
- ModelNet40 partial-to-partial registration
- Robust to arbitrary SE(3) transformations
- Up to 4096 points per cloud
"""

import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.ModelNet40Dataset,
        'args': {
            'data_root': 'data/datasets/soft_links/ModelNet40',
            'split': 'train',
            'dataset_size': 10000,  # Standard training size
            'keep_ratio': 0.7,  # Partial point clouds (70% of points)
            'rotation_mag': 45.0,  # Random rotation up to 45 degrees
            'translation_mag': 0.5,  # Random translation up to 0.5 units
            'matching_radius': 0.05,  # Radius for correspondence finding
            'overlap_range': (0.3, 1.0),  # Overlap range for filtering
            'min_points': 512,  # Minimum points filter
            'cache_filepath': 'data/datasets/soft_links/ModelNet40_gmcnet_train_cache.json',
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        {
                            'op': {
                                'class': data.transforms.vision_3d.Clamp,
                                'args': {'max_points': 4096},  # Limit to 4096 points
                            },
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
                        {
                            'op': {
                                'class': data.transforms.vision_3d.RandomPointCrop,
                                'args': {
                                    'keep_ratio': 0.7,  # Partial point clouds
                                },
                            },
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
                        {
                            'op': {
                                'class': data.transforms.vision_3d.GaussianPosNoise,
                                'args': {'std': 0.01},  # Add noise for robustness
                            },
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
                    ],
                },
            },
        },
    },
    'train_dataloader': {
        'class': data.dataloaders.BaseDataLoader,
        'args': {
            'batch_size': 1,  # GMCNet typically uses batch_size=1
            'num_workers': 4,
            'shuffle': True,
        },
    },
}