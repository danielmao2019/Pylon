import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.BiTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'cache_dirname': 'bitemporal_pcr_cache',
            'split': 'train',
            'dataset_size': 5000,  # Total number of synthetic registration pairs to generate
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
                        # PCR translation normalization
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
