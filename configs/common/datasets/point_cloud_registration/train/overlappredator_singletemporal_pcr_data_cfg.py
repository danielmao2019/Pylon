import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.SynthPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'cache_dirname': 'synth_pcr_cache',
            'split': 'train',
            'dataset_size': 5000,  # Total number of synthetic registration pairs to generate
            'overlap_range': (0.0, 1.0),  # GeoTransformer doesn't use specific overlap ranges
            'matching_radius': 0.05,  # Radius for correspondence finding
            'rotation_mag': 45.0,  # GeoTransformer synthetic transform parameters
            'translation_mag': 0.5,  # GeoTransformer synthetic transform parameters
            'voxel_size': 10.0,
            'min_points': 512,
            'max_points': 8192,
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        # OverlapPredator cropping for creating overlaps after synthetic transform
                        (
                            {
                                'class': data.transforms.vision_3d.RandomPointCrop,
                                'args': {'keep_ratio': 0.7},
                            },
                            [('inputs', 'src_pc')],
                        ),
                    ],
                },
            },
        },
    },
    'train_dataloader': {
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
    'criterion': None,
}
