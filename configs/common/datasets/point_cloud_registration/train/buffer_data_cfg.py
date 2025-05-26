import torch
import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'train',
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        {
                            'op': data.transforms.vision_3d.UniformPosNoise(min=-0.5*0.05, max=+0.5*0.05),
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
                        {
                            'op': data.transforms.vision_3d.DownSample(voxel_size=0.05),
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
                        {
                            'op': data.transforms.vision_3d.Shuffle(),
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
                        {
                            'op': data.transforms.vision_3d.RandomRigidTransform(rot_mag=180.0, trans_mag=0.0),
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')],
                        },
                        {
                            'op': data.transforms.vision_3d.DownSample(voxel_size=0.30),
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
                        {
                            'op': data.transforms.vision_3d.Shuffle(),
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
                        {
                            'op': data.transforms.vision_3d.Clamp(max_points=40000),
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
                    ],
                },
            },
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
        },
    },
    'criterion': None,
}
