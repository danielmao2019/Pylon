import torch
import data
from metrics.vision_3d import RegistrationRecall


data_cfg = {
    'eval_dataset': {
        'class': data.datasets.RealPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'split': 'val',
            'voxel_size': 10.0,
            'min_points': 256,
            'max_points': 8192,
            'device': 'cpu',
            'transforms_cfg': {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        (
                            data.transforms.vision_3d.PCRTranslation(),
                            [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')],
                        ),
                    ],
                },
            },
        },
    },
    'eval_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
        },
    },
    'metric': {
        'class': RegistrationRecall,
        'args': {
            'rot_threshold_deg': 5.0,
            'trans_threshold_m': 0.3,
        },
    },
}
