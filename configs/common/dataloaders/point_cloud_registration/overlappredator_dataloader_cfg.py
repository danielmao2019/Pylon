import data


train_dataloader_cfg = {
    'class': data.dataloaders.OverlapPredatorDataloader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'shuffle': True,
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
}

val_dataloader_cfg = {
    'class': data.dataloaders.OverlapPredatorDataloader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'shuffle': False,
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
}
