import data


data_cfg = {
    'eval_dataset': {
        'class': data.datasets.ModelNet40Dataset,
        'args': {
            'data_root': 'data/datasets/soft_links/ModelNet40',
            'split': 'test',
            'dataset_size': 500,  # Smaller dataset size for evaluation
            'rotation_mag': 45.0,  # Synthetic transform parameters
            'translation_mag': 0.5,  # Synthetic transform parameters
            'matching_radius': 0.05,  # Radius for correspondence finding
            'overlap_range': (0.3, 1.0),
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
