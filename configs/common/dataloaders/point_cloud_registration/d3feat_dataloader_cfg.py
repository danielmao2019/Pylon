import data


train_dataloader_cfg = {
    'class': data.dataloaders.D3FeatDataLoader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'shuffle': True,
        'config': {
            'num_layers': 5,
            'first_subsampling_dl': 0.03,
            'conv_radius': 2.5,
            'deform_radius': 5.0,
            'num_kernel_points': 15,
        },
    },
}

val_dataloader_cfg = {
    'class': data.dataloaders.D3FeatDataLoader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'shuffle': False,
        'config': {
            'num_layers': 5,
            'first_subsampling_dl': 0.03,
            'conv_radius': 2.5,
            'deform_radius': 5.0,
            'num_kernel_points': 15,
        },
    },
}
