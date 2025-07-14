import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.SingleTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'cache_dirname': 'single_temporal_pcr_cache',
            'split': 'train',
            'dataset_size': 5000,  # Total number of synthetic registration pairs to generate
            'overlap_range': (0.0, 1.0),  # GeoTransformer doesn't use specific overlap ranges
            'matching_radius': 0.05,  # Radius for correspondence finding
            'rotation_mag': 45.0,  # GeoTransformer synthetic transform parameters
            'translation_mag': 0.5,  # GeoTransformer synthetic transform parameters
            'min_points': 512,  # Minimum points filter for cache generation
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        (
                            {
                                'class': data.transforms.vision_3d.Clamp,
                                'args': {'max_points': 8192},
                            },
                            [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        ),
                    ],
                },
            },
        },
    },
    'train_dataloader': {
        'class': data.dataloaders.GeoTransformerDataloader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'num_stages': 4,
            'voxel_size': 0.1,
            'search_radius': 2.5 * 0.1,
        },
    },
    'criterion': None,
}
