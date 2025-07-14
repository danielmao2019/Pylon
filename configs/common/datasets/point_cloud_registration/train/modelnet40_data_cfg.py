import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.ModelNet40Dataset,
        'args': {
            'data_root': 'data/datasets/soft_links/ModelNet40',
            'split': 'train',
            'dataset_size': 50,  # Small size for viewer testing
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
                                'args': {'min_points': 512, 'max_points': 4096},
                            },
                            [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        ),
                    ],
                },
            },
        },
    },
}
