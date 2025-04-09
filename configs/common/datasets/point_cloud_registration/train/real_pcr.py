from data.datasets import RealPCRDataset


data_cfg = {
    'train_dataset': {
        'class': RealPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'split': 'train',
            'voxel_size': 10.0,
            'min_points': 256,
            'max_points': 8192,
        },
    },
}
