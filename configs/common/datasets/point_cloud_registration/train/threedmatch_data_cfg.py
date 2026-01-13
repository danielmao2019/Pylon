from functools import partial
import data
import utils
from data.structures.three_d.point_cloud.ops.correspondences import get_correspondences


data_cfg = {
    'train_dataset': {
        'class': data.datasets.ThreeDMatchDataset,
        'args': {
            'data_root': './data/datasets/soft_links/threedmatch',
            'split': 'train',
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
                        {
                            'op': partial(get_correspondences, radius=0.0375),
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')],
                            'output_names': [('inputs', 'correspondences')],
                        },
                        (
                            {
                                'class': data.transforms.vision_3d.RandomRigidTransform,
                                'args': {
                                    'rot_mag': 45.0,
                                    'trans_mag': 0.5,
                                    'method': 'Rodrigues',
                                },
                            },
                            [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')],
                        ),
                        (
                            {
                                'class': data.transforms.vision_3d.GaussianPosNoise,
                                'args': {
                                    'std': 0.005,
                                },
                            },
                            [('inputs', 'src_pc')],
                        ),
                        (
                            {
                                'class': data.transforms.vision_3d.GaussianPosNoise,
                                'args': {
                                    'std': 0.005,
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
