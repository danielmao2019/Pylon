import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.ModelNet40Dataset,
        'args': {
            'data_root': 'data/datasets/soft_links/ModelNet40',
            'split': 'train',
            'dataset_size': 50,  # Small size for viewer testing
            'overlap_range': (0.3, 1.0),  # GeoTransformer range with cropping
            'matching_radius': 0.05,  # Radius for correspondence finding
            'rotation_mag': 45.0,  # GeoTransformer synthetic transform parameters
            'translation_mag': 0.5,  # GeoTransformer synthetic transform parameters
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        # SE(3) transformation for data augmentation (separate from synthetic transforms)
                        (
                            {
                                'class': data.transforms.vision_3d.RandomRigidTransform,
                                'args': {'rot_mag': 45.0, 'trans_mag': 0.5},
                            },
                            [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')],
                        ),
                        # GeoTransformer cropping for creating overlaps after synthetic transform
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
