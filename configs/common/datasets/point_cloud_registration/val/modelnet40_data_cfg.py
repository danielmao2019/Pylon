import data


data_cfg = {
    'train_dataset': {  # Note: viewer always looks for 'train_dataset' key even for val configs
        'class': data.datasets.ModelNet40Dataset,
        'args': {
            'data_root': 'data/datasets/soft_links/ModelNet40',
            'split': 'test',  # Use test split for validation
            'dataset_size': 500,  # Smaller dataset size for validation
            'overlap_range': (0.4, 0.8),  # Narrower overlap range for consistent validation
            'matching_radius': 0.05,  # Same radius as training
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