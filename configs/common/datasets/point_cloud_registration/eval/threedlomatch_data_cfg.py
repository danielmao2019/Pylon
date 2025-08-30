import data


data_cfg = {
    'eval_dataset': {
        'class': data.datasets.ThreeDLoMatchDataset,
        'args': {
            'data_root': './data/datasets/soft_links/threedmatch',
            'split': 'test',
            'matching_radius': 0.1,
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [],
                },
            },
        },
    },
}