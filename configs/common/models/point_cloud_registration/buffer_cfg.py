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
                'pose_refine': False,
            },
            'point': {
                'keypts_th': 0.5,
                'num_keypts': 1500,
            },
            'patch': {
                'azi_n': 20,
                'ele_n': 7,
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
