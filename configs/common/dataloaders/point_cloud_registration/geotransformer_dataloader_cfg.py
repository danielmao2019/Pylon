import data


train_dataloader_cfg = {
    'class': data.dataloaders.GeoTransformerDataloader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'shuffle': True,
        'num_stages': 4,
        'voxel_size': 0.1,
        'search_radius': 2.5 * 0.1,
    },
}

val_dataloader_cfg = {
    'class': data.dataloaders.GeoTransformerDataloader,
    'args': {
        'batch_size': 1,
        'num_workers': 4,
        'shuffle': False,
        'num_stages': 4,
        'voxel_size': 0.1,
        'search_radius': 2.5 * 0.1,
    },
}
