import data


data_cfg = {
    'val_dataset': {
        'class': data.datasets.BiTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'cache_filepath': './data/datasets/soft_links/ivision-pcr-data/../bi_temporal_pcr_cache.json',
            'split': 'val',
            'dataset_size': 1000,  # Smaller for validation
            'rotation_mag': 45.0,  # GeoTransformer synthetic transform parameters
            'translation_mag': 0.5,  # GeoTransformer synthetic transform parameters
            'matching_radius': 0.05,  # Radius for correspondence finding
            'overlap_range': (0.0, 1.0),  # GeoTransformer doesn't use specific overlap ranges
            'min_points': 512,  # Minimum points filter for cache generation
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
    'val_dataloader': {
        'class': data.dataloaders.OverlapPredatorDataloader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'config': {
                'deform_radius': 5.0,
                'num_layers': 4,
                'first_subsampling_dl': 0.3,
                'conv_radius': 4.25,
                'architecture': [
                    'simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'last_unary',
                ],
            },
        },
    },
    'metric': None,
}
