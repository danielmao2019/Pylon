import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.RealPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'split': 'train',
            'gt_transforms': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'voxel_size': 10.0,
            'min_points': 512,
            'max_points': 8192,
            'transform': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        (
                            data.transforms.vision_3d.PCRTranslation,
                            [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')],
                        ),
                    ],
                },
            },
        },
    },
    'train_dataloader': {
        'class': data.dataloaders.GeoTransformerDataloader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'num_stages': 4,
            'voxel_size': 0.1,
            'search_radius': 2.5 * 0.1,
        },
    },
    'criterion': None,
}
