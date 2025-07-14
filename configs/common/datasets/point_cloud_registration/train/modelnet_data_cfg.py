import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.ModelNetDataset,
        'args': {
            'data_root': 'data/datasets/soft_links/ModelNet40',
            'split': 'train',
            'dataset_size': 1000,  # Total number of synthetic registration pairs
            'overlap_range': (0.3, 1.0),  # Overlap range for generated pairs
            'matching_radius': 0.05,  # Radius for correspondence finding
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        # SE(3) transformation with random rotation and translation
                        (
                            {
                                'class': data.transforms.vision_3d.RandomRigidTransform,
                                'args': {'rot_mag': 45.0, 'trans_mag': 0.5},
                            },
                            [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')],
                        ),
                        # Random cropping for source point cloud (plane-based by default)
                        (
                            {
                                'class': data.transforms.vision_3d.RandomPlaneCrop,
                                'args': {'keep_ratio': 0.7},
                            },
                            [('inputs', 'src_pc')],
                        ),
                    ],
                },
            },
        },
    },
}