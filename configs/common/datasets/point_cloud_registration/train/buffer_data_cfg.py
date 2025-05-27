import easydict as edict
import data
import criteria


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
                            'output_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
                        },
                        {
                            'op': data.transforms.vision_3d.Shuffle(),
                            'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
                        },
                        {
                            'op': data.transforms.vision_3d.RandomRigidTransform(rot_mag=180.0, trans_mag=0.0),
                            'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'), ('labels', 'transform')],
                        },
                        {
                            'op': data.transforms.vision_3d.DownSample(voxel_size=0.30),
                            'input_names': [('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds')],
                            'output_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
                        },
                        {
                            'op': data.transforms.vision_3d.Shuffle(),
                            'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
                        },
                        {
                            'op': data.transforms.vision_3d.Clamp(max_points=40000),
                            'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
                        },
                        {
                            'op': data.transforms.vision_3d.EstimateNormals(),
                            'input_names': [('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds')],
                        },
                        {
                            'op': data.transforms.Identity(),
                            'input_names': [
                                ('inputs', 'src_pc_fds'), ('inputs', 'tgt_pc_fds'),
                                ('inputs', 'src_pc_sds'), ('inputs', 'tgt_pc_sds'),
                                ('labels', 'transform'),
                                ('meta_info', 'seq'), ('meta_info', 't0'), ('meta_info', 't1'),
                            ],
                        },
                    ],
                },
            },
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
    'criterion': None,
}

multi_stage_criterion_cfg = [
    {
        'class': criteria.vision_3d.point_cloud_registration.BUFFER_RefStageCriterion,
        'args': {},
    },
    {
        'class': criteria.vision_3d.point_cloud_registration.BUFFER_DescStageCriterion,
        'args': {},
    },
    {
        'class': criteria.vision_3d.point_cloud_registration.BUFFER_KeyptStageCriterion,
        'args': {},
    },
    {
        'class': criteria.vision_3d.point_cloud_registration.BUFFER_InlierStageCriterion,
        'args': {},
    },
]
