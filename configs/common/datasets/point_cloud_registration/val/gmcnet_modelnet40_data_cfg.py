"""
GMCNet validation dataset configuration for ModelNet40.

This config provides validation data for GMCNet training with simpler transforms
to ensure consistent evaluation.
"""

import data


data_cfg = {
    'val_dataset': {
        'class': data.datasets.ModelNet40Dataset,
        'args': {
            'data_root': 'data/datasets/soft_links/ModelNet40',
            'split': 'test',  # Use test split for validation
            'dataset_size': 1000,  # Smaller validation set
            'keep_ratio': 0.7,  # Partial point clouds (70% of points)
            'rotation_mag': 45.0,  # Random rotation up to 45 degrees
            'translation_mag': 0.5,  # Random translation up to 0.5 units
            'matching_radius': 0.05,  # Radius for correspondence finding
            'overlap_range': (0.3, 1.0),  # Overlap range for filtering
            'min_points': 512,  # Minimum points filter
            'cache_filepath': 'data/datasets/soft_links/ModelNet40_gmcnet_val_cache.json',
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
                        # No noise for validation to ensure consistent evaluation
                    ],
                },
            },
        },
    },
    'val_dataloader': {
        'class': data.dataloaders.BaseDataLoader,
        'args': {
            'batch_size': 1,  # Always batch_size=1 for validation
            'num_workers': 4,
            'shuffle': False,
        },
    },
}