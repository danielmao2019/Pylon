import data


train_dataloader_cfg = {
    'class': data.dataloaders.PARENetDataloader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'shuffle': True,
        'num_stages': 4,
        'voxel_size': 0.05,
        'subsample_ratio': 4.0,
        'num_neighbors': [32, 32, 32, 32],
        'precompute_data': True,
    },
}

val_dataloader_cfg = {
    'class': data.dataloaders.PARENetDataloader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'shuffle': False,
        'num_stages': 4,
        'voxel_size': 0.05,
        'subsample_ratio': 4.0,
        'num_neighbors': [32, 32, 32, 32],
        'precompute_data': True,
    },
}
