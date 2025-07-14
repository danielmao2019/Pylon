import data


data_cfg = {
    'val_dataset': {
        'class': data.datasets.SingleTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'cache_dirname': 'singletemporal_pcr_cache',
            'split': 'val',
            'dataset_size': 1000,  # Smaller for validation
            'overlap_range': (0.0, 1.0),  # GeoTransformer doesn't use specific overlap ranges
            'matching_radius': 0.05,  # Radius for correspondence finding
            'rotation_mag': 45.0,  # GeoTransformer synthetic transform parameters
            'translation_mag': 0.5,  # GeoTransformer synthetic transform parameters
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        # Point sampling transform - moved from dataset initialization
                        (
                            {
                                'class': data.transforms.vision_3d.RandomPointSampling,
                                'args': {'min_points': 512, 'max_points': 8192},
                            },
                            [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        ),
                    ],
                },
            },
        },
    },
    'val_dataloader': {
        'class': data.dataloaders.GeoTransformerDataloader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'num_stages': 4,
            'voxel_size': 0.1,
            'search_radius': 2.5 * 0.1,
        },
    },
    'metric': None,
}
