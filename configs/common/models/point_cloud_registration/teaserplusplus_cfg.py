from models.point_cloud_registration.classic.teaserplusplus import TeaserPlusPlus

model_cfg = {
    'class': TeaserPlusPlus,
    'args': {
        'estimate_rotation': False,
        'estimate_scaling': False,
        'correspondences': 'fpfh',
        'voxel_size': 0.05,
    },
}
