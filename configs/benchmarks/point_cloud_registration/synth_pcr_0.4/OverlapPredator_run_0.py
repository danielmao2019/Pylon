# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from criteria.vision_3d.point_cloud_registration.overlappredator_criterion.overlappredator_criterion import OverlapPredatorCriterion
from data.dataloaders.overlappredator_dataloader import OverlapPredatorDataloader
from data.datasets.pcr_datasets.synth_pcr_dataset import SynthPCRDataset
from data.transforms.compose import Compose
from data.transforms.vision_3d.random_rigid_transform import RandomRigidTransform
from metrics.vision_3d.point_cloud_registration.overlappredator_metric.overlappredator_metric import OverlapPredatorMetric
from models.point_cloud_registration.overlappredator.overlappredator import OverlapPredator
from optimizers.single_task_optimizer import SingleTaskOptimizer
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer


config = {
    'runner': SupervisedSingleTaskTrainer,
    'work_dir': './logs/benchmarks/point_cloud_registration/synth_pcr_0.4/OverlapPredator_run_0',
    'epochs': 100,
    'init_seed': 7980958,
    'train_seeds': [84519219, 93329737, 77147470, 1918058, 4607827, 67541655, 63812609, 60961990, 65765005, 66269223, 4751528, 60394330, 17129707, 59193350, 36673206, 94205849, 5383100, 4940629, 26023637, 68790284, 34521817, 45107714, 78894, 93371983, 62919426, 13619828, 74051870, 88891079, 99599740, 98814980, 18633816, 95897386, 69213983, 21297709, 62674748, 10155765, 2338891, 79584973, 34641582, 9739540, 21031458, 61665830, 21519089, 32718300, 62260867, 4505950, 52953630, 42019328, 18688623, 90918524, 6624703, 35229539, 18785807, 86740117, 23565079, 45908564, 17737955, 55429313, 18201128, 92817139, 26932871, 94543626, 48683864, 40094803, 9253583, 50790271, 30730495, 92022337, 11253113, 82861573, 46678341, 79280654, 27757198, 10818100, 11982493, 52485555, 6267839, 65775789, 84566929, 62102393, 2338468, 63759415, 48933155, 66649749, 38680260, 66056099, 83459747, 24518241, 54862005, 61033894, 12160826, 63207538, 57381803, 24172428, 14353360, 91701504, 65872644, 83888385, 65714224, 91822362],
    'val_seeds': [44451211, 46973892, 91980466, 10542869, 90778031, 15241770, 85834391, 301858, 35979474, 97667037, 38913680, 16479667, 14147194, 22583175, 24142189, 3989770, 38355646, 41757981, 59305907, 51527696, 20412509, 17085532, 72065017, 78788244, 17460257, 13488487, 52416318, 26292699, 91080980, 62391926, 38797993, 55350203, 25266115, 6787797, 7262893, 76275785, 81997438, 15690902, 77974184, 38837898, 213978, 82613509, 3430388, 44178109, 14550901, 52287208, 78465776, 44630622, 26314105, 53123909, 78288927, 65762064, 86989183, 65363567, 11457327, 32266074, 98131990, 94239429, 74902634, 41823403, 44858388, 79003591, 46863258, 96155470, 27793230, 95025018, 76054504, 24828831, 58627503, 24085536, 63032156, 93369974, 97846568, 8633720, 43410773, 5997087, 20658439, 15355830, 53724678, 19385519, 6483198, 74320022, 26252752, 56345032, 7557386, 83107327, 43756610, 62286111, 33992207, 32397448, 7027956, 73666772, 75157953, 80826683, 33170978, 55334611, 91812956, 12663603, 3738398, 11190218],
    'test_seed': 76296785,
    'train_dataset': {
        'class': SynthPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'cache_dirname': 'synth_pcr_cache',
            'split': 'train',
            'voxel_size': 10.0,
            'min_points': 512,
            'max_points': 8192,
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [(
    {
            'class': RandomRigidTransform,
            'args': {
                'rot_mag': 45.0,
                'trans_mag': 0.5,
            },
        },
    [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')]
)],
                },
            },
            'overlap': 0.4,
        },
    },
    'train_dataloader': {
        'class': OverlapPredatorDataloader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'config': {
                'deform_radius': 5.0,
                'num_layers': 4,
                'first_subsampling_dl': 0.3,
                'conv_radius': 4.25,
                'architecture': ['simple', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb', 'nearest_upsample', 'unary', 'nearest_upsample', 'unary', 'nearest_upsample', 'last_unary'],
            },
        },
    },
    'criterion': {
        'class': OverlapPredatorCriterion,
        'args': {
            'log_scale': 48,
            'pos_optimal': 0.1,
            'neg_optimal': 1.4,
            'pos_margin': 0.1,
            'neg_margin': 1.4,
            'max_points': 512,
            'safe_radius': 0.75,
            'matchability_radius': 0.3,
            'pos_radius': 0.21,
            'w_circle_loss': 1.0,
            'w_overlap_loss': 1.0,
            'w_saliency_loss': 0.0,
        },
    },
    'val_dataset': {
        'class': SynthPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'cache_dirname': 'synth_pcr_cache',
            'split': 'val',
            'voxel_size': 10.0,
            'min_points': 512,
            'max_points': 8192,
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [(
    {
            'class': RandomRigidTransform,
            'args': {
                'rot_mag': 45.0,
                'trans_mag': 0.5,
            },
        },
    [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')]
)],
                },
            },
            'overlap': 0.4,
        },
    },
    'val_dataloader': {
        'class': OverlapPredatorDataloader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'config': {
                'deform_radius': 5.0,
                'num_layers': 4,
                'first_subsampling_dl': 0.3,
                'conv_radius': 4.25,
                'architecture': ['simple', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb', 'nearest_upsample', 'unary', 'nearest_upsample', 'unary', 'nearest_upsample', 'last_unary'],
            },
        },
    },
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': OverlapPredatorMetric,
        'args': {
            'max_points': 512,
            'matchability_radius': 0.3,
            'pos_radius': 0.21,
        },
    },
    'model': {
        'class': OverlapPredator,
        'args': {
            'architecture': ['simple', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb', 'nearest_upsample', 'unary', 'nearest_upsample', 'unary', 'nearest_upsample', 'last_unary'],
            'num_layers': 4,
            'in_points_dim': 3,
            'first_feats_dim': 256,
            'final_feats_dim': 32,
            'first_subsampling_dl': 0.3,
            'in_feats_dim': 1,
            'conv_radius': 4.25,
            'deform_radius': 5.0,
            'num_kernel_points': 15,
            'KP_extent': 2.0,
            'KP_influence': 'linear',
            'aggregation_mode': 'sum',
            'fixed_kernel_points': 'center',
            'use_batch_norm': True,
            'batch_norm_momentum': 0.02,
            'deformable': False,
            'modulated': False,
            'add_cross_score': True,
            'condition_feature': True,
            'gnn_feats_dim': 256,
            'dgcnn_k': 10,
            'num_head': 4,
            'nets': ['self', 'cross', 'self'],
        },
    },
    'optimizer': {
        'class': SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': Adam,
                'args': {
                    'params': None,
                    'lr': 0.0001,
                    'weight_decay': 1e-06,
                },
            },
        },
    },
    'scheduler': {
        'class': StepLR,
        'args': {
            'optimizer': None,
            'step_size': 1000,
            'gamma': 0.95,
        },
    },
}
