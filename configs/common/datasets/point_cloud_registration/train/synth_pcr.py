from data.datasets import SynthPCRDataset


data_cfg = {
    'train_dataset': {
        'class': SynthPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'split': 'train',
            'voxel_size': 10.0,
            'min_points': 256,
            'max_points': 8192,
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        (
                            data.transforms.vision_3d.RandomRigidTransform(rot_mag=45.0, trans_mag=0.5),
                            [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')],
                        ),
                    ],
                },
            },
        },
    },
}
