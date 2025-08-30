import data
from configs.common.datasets.point_cloud_registration.train.kitti_data_cfg import get_kitti_transforms


data_cfg = {
    'val_dataset': {
        'class': data.datasets.KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'val',
            'transforms_cfg': get_kitti_transforms('Euler', 3),
        },
    },
}
