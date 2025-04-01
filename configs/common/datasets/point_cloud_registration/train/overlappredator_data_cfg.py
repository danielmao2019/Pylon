import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.SynthPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'split': 'train',
            'rot_mag': 45.0,
            'trans_mag': 0.5,
            'voxel_size': 5.0,
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
