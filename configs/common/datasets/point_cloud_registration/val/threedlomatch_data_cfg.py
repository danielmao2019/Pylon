import data
import utils


data_cfg = {
    'val_dataset': {
        'class': data.datasets.ThreeDLoMatchDataset,
        'args': {
            'data_root': './data/datasets/soft_links/threedmatch',
            'split': 'val',
            'matching_radius': 0.1,
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        (
                            {
                                'class': data.structures.three_d.point_cloud.ops.RandomSelect,
                                'args': {
                                    'count': 5000,
                                },
                            },
                            [('inputs', 'src_pc')],
                        ),
                        (
                            {
                                'class': data.structures.three_d.point_cloud.ops.RandomSelect,
                                'args': {
                                    'count': 5000,
                                },
                            },
                            [('inputs', 'tgt_pc')],
                        ),
                    ],
                },
            },
        },
    },
}