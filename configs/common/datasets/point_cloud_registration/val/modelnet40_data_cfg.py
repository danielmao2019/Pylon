import data


data_cfg = {
    'train_dataset': {  # Note: viewer always looks for 'train_dataset' key even for val configs
        'class': data.datasets.ModelNet40Dataset,
        'args': {
            'data_root': 'data/datasets/soft_links/ModelNet40',
            'split': 'test',  # Use test split for validation
            'dataset_size': 500,  # Smaller dataset size for validation
            'overlap_range': (0.0, 1.0),  # GeoTransformer doesn't use specific overlap ranges
            'matching_radius': 0.05,  # Radius for correspondence finding
            'rotation_mag': 45.0,  # GeoTransformer synthetic transform parameters
            'translation_mag': 0.5,  # GeoTransformer synthetic transform parameters
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        # Random cropping with consistent ratio
                        (
                            {
                                'class': data.transforms.vision_3d.RandomPlaneCrop,
                                'args': {'keep_ratio': 0.7},
                            },
                            [('inputs', 'src_pc')],
                        ),
                    ],
                },
            },
        },
    },
}