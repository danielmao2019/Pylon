import torch
import data


data_cfg = {
    'val_dataset': {
        'class': data.datasets.KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'val',
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
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
        },
    },
    'testdataset': {
        'class': data.datasets.KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'test',
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        {
                            'op': data.transforms.vision_3d.DownSample(voxel_size=0.05),
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
                        {
                            'op': data.transforms.vision_3d.Shuffle(),
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
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
                        {
                            'op': data.transforms.vision_3d.EstimateNormals(),
                            'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
                        },
                    ],
                },
            },
        },
    },
    'test_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
        },
    },
    'metric': None,
}
