from easydict import EasyDict as edict
from models.point_cloud_registration import BUFFER


model_cfg = {
    'class': BUFFER,
    'args': {
        'config': edict({
            'data': {
                'voxel_size_0': 0.30,
            },
            'train': {
                'pos_num': 512,
            },
            'test': {
                'scale': 0.30 / 0.30,
                'pose_refine': False,
            },
            'point': {
                'in_feats_dim': 3,
                'first_feats_dim': 32,
                'conv_radius': 2.0,
                'keypts_th': 0.5,
                'num_keypts': 1500,
            },
            'patch': {
                'des_r': 3.0,
                'num_points_per_patch': 512,
                'rad_n': 3,
                'azi_n': 20,
                'ele_n': 7,
                'delta': 0.8,
                'voxel_sample': 10,
            },
            'match': {
                'dist_th': 0.30,
                'inlier_th': 2.0,
                'similar_th': 0.9,
                'confidence': 1.0,
                'iter_n': 50000,
            },
        }),
    },
}
