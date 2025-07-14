import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.BiTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'cache_dirname': 'bi_temporal_pcr_cache',
            'split': 'train',
            'dataset_size': 5000,  # Total number of synthetic registration pairs to generate
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
                        # PCR translation normalization
                        {
                            'op': {
                                'class': data.transforms.vision_3d.PCRTranslation,
                                'args': {},
                            },
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')],
                        },
                    ],
                },
            },
        },
    },
}
