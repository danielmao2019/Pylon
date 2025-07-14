import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.BiTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'cache_dirname': 'real_pcr_cache',
            'split': 'train',
            'dataset_size': 5000,  # Total number of synthetic registration pairs to generate
            'overlap_range': (0.0, 1.0),  # GeoTransformer doesn't use specific overlap ranges
            'matching_radius': 0.05,  # Radius for correspondence finding
            'rotation_mag': 45.0,  # GeoTransformer synthetic transform parameters
            'translation_mag': 0.5,  # GeoTransformer synthetic transform parameters
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
}
