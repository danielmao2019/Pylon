import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.ModelNet40Dataset,
        'args': {
            'data_root': 'data/datasets/soft_links/ModelNet40',
            'split': 'train',
            'dataset_size': 50,  # Small size for viewer testing
            'rotation_mag': 45.0,  # GeoTransformer synthetic transform parameters
            'translation_mag': 0.5,  # GeoTransformer synthetic transform parameters
            'matching_radius': 0.05,  # Radius for correspondence finding
            'overlap_range': (0.0, 1.0),  # GeoTransformer doesn't use specific overlap ranges
            'min_points': 512,  # Minimum points filter for cache generation
            'cache_filepath': 'data/datasets/soft_links/ModelNet40/../ModelNet40_cache.json',
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        {
                            'op': {
                                'class': data.transforms.vision_3d.Clamp,
                                'args': {'max_points': 4096},
                            },
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
                    ],
                },
            },
        },
    },
}
