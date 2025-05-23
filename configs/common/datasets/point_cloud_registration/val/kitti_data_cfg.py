import data


data_cfg = {
    'val_dataset': {
        'class': data.datasets.KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/kitti',
            'split': 'val',
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
        },
    },
    'metric': None,
}
