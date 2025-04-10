import torch
import data
from metrics.vision_3d import RegistrationRecall


data_cfg = {
    'eval_dataset': {
        'class': data.datasets.SynthPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'cache_dirname': 'synth_pcr_cache',
            'split': 'val',
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
            'device': 'cpu',
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
