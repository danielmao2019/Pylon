import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'train',
        },
    },
    'train_dataloader': {
        'class': data.dataloaders.PARENetDataloader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'num_stages': 4,
            'voxel_size': 0.05,
            'subsample_ratio': 4.0,
            'num_neighbors': [32, 32, 32, 32],
            'precompute_data': True,
        },
    },
    'criterion': None,
}
