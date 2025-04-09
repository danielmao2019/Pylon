from data.datasets import RealPCRDataset


data_cfg = {
    'train_dataset': {
        'class': RealPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'split': 'train',
            'rot_mag': 45.0,
            'trans_mag': 0.5,
            'voxel_size': 10.0,
            'min_points': 256,
            'max_points': 8192,
        },
    },
}
