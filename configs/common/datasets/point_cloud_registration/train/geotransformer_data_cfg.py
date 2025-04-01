import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.SynthPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'split': 'train',
            'rot_mag': 45.0,
            'trans_mag': 0.5,
            'voxel_size': 1.0,
        },
    },
    'train_dataloader': {
        'class': data.dataloaders.GeoTransformerDataloader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'num_stages': 4,
            'voxel_size': 0.025,
            'search_radius': 2.5 * 0.025,
        },
    },
    'criterion': None,
}
