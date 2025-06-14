import easydict as edict
import data


def get_transforms_cfg(method: str, num_axis: int) -> dict:
    return {
        'class': data.transforms.Compose,
        'args': {
            'transforms': [
                {
                    'op': {
                        'class': data.transforms.vision_3d.UniformPosNoise,
                        'args': {'min': -0.5*0.05, 'max': +0.5*0.05},
                    },
                    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                },
                {
                    'op': {
                        'class': data.transforms.vision_3d.DownSample,
                        'args': {'voxel_size': 0.05},
                    },
                    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                    'output_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
                },
                {
                    'op': {
                        'class': data.transforms.vision_3d.Shuffle,
                        'args': {},
                    },
                    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
                },
                {
                    'op': {
                        'class': data.transforms.vision_3d.RandomRigidTransform,
                        'args': {'rot_mag': 180.0, 'trans_mag': 0.0, 'method': method, 'num_axis': num_axis},
                    },
                    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('labels', 'transform')],
                },
                {
                    'op': {
                        'class': data.transforms.vision_3d.DownSample,
                        'args': {'voxel_size': 0.30},
                    },
                    'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
                    'output_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
                },
                {
                    'op': {
                        'class': data.transforms.vision_3d.Shuffle,
                        'args': {},
                    },
                    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
                },
                {
                    'op': {
                        'class': data.transforms.vision_3d.Clamp,
                        'args': {'max_points': 40000},
                    },
                    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
                },
                {
                    'op': {
                        'class': data.transforms.vision_3d.EstimateNormals,
                        'args': {},
                    },
                    'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
                },
                {
                    'op': {
                        'class': data.transforms.Identity,
                        'args': {},
                    },
                    'input_names': [
                        ('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'),
                        ('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds'),
                        ('labels', 'transform'),
                        ('meta_info', 'seq'), ('meta_info', 't0'), ('meta_info', 't1'),
                    ],
                },
            ],
        },
    }


data_cfg = {
    'train_dataset': {
        'class': data.datasets.KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'train',
            'transforms_cfg': None,
        },
    },
    'train_dataloader': {
        'class': data.dataloaders.BufferDataloader,
        'args': {
            'config': edict.EasyDict({
                'point': {
                    'conv_radius': 2.0,
                },
                'data': {
                    'voxel_size_0': 0.30,
                },
            }),
            'batch_size': 1,
        },
    },
}
