# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from criteria.vision_3d.point_cloud_registration.geotransformer_criterion.geotransformer_criterion import GeoTransformerCriterion
from data.dataloaders.geotransformer_dataloader import GeoTransformerDataloader
from data.datasets.pcr_datasets.bi_temporal_pcr_dataset import BiTemporalPCRDataset
from data.transforms.compose import Compose
from data.transforms.vision_3d.clamp import Clamp
from metrics.vision_3d.point_cloud_registration.geotransformer_metric.geotransformer_metric import GeoTransformerMetric
from models.point_cloud_registration.geotransformer.geotransformer import GeoTransformer
from optimizers.single_task_optimizer import SingleTaskOptimizer
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer


config = {
    'runner': SupervisedSingleTaskTrainer,
    'work_dir': './logs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.5/GeoTransformer_run_2',
    'epochs': 100,
    'init_seed': 89069464,
    'train_seeds': [97133341, 25479656, 39743726, 1210687, 55528779, 95962679, 32256474, 85437011, 47915760, 28395322, 82321529, 87317667, 53228815, 94844360, 27787891, 54525464, 25315354, 82692950, 90772293, 27807892, 77672973, 51694077, 25881336, 97804637, 579121, 90179973, 33546895, 27852764, 86843228, 98264527, 57549261, 58559623, 95221740, 66101580, 22287143, 43345023, 14868914, 2376299, 20709559, 40298619, 6784968, 45319587, 26796622, 64043978, 63294971, 690339, 94597830, 82560351, 97441980, 80191454, 1139460, 96269961, 69959142, 44467027, 2988153, 75389050, 13148724, 52354631, 42961934, 27852045, 48702467, 94155663, 75003897, 23384003, 45811414, 78049903, 1353167, 22667520, 65961871, 20194725, 26809878, 31693973, 64501519, 83592181, 82977390, 99568177, 61608666, 62505429, 32562073, 42596997, 50825703, 1930446, 72029576, 66214698, 26329525, 89802938, 37973934, 79215595, 65353728, 62391848, 63932999, 27321482, 6692801, 71241660, 70467989, 20230491, 83900716, 45928872, 71411593, 63700826],
    'val_seeds': [65258311, 46287792, 59400215, 26279264, 91494731, 52196190, 34239288, 92691512, 57732367, 60516337, 72538414, 72660144, 83383320, 44427507, 68295412, 94527899, 41224654, 9004140, 88827576, 50282840, 91829756, 94153178, 28411656, 4358843, 53893264, 20252494, 32299244, 42086114, 40709365, 68374976, 5154854, 76465417, 4806468, 81067151, 54722971, 82273225, 7136812, 11791738, 38545632, 30496711, 25146057, 47854043, 82948225, 63300967, 19259000, 15882049, 81496504, 19660963, 73194625, 88699978, 93866547, 58021462, 13475621, 72617727, 41032482, 18765016, 71681362, 7870529, 20620576, 50212577, 48624899, 55059889, 11890433, 92561082, 57982682, 82485150, 60098375, 46811022, 36087785, 39676787, 11536842, 60002617, 72796153, 43721643, 21558202, 50137401, 92094726, 21923066, 7897250, 82342366, 17169442, 48271109, 44226056, 13136235, 77066728, 35383095, 22833625, 83116895, 23965410, 13216055, 25624591, 11453896, 33782625, 22377355, 96209414, 41126830, 39802664, 35932666, 78424634, 93157574],
    'test_seed': 56630915,
    'train_dataset': {
        'class': BiTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'cache_filepath': './data/datasets/soft_links/ivision-pcr-data/../bi_temporal_pcr_cache.json',
            'split': 'train',
            'dataset_size': 5000,
            'rotation_mag': 45.0,
            'translation_mag': 0.5,
            'matching_radius': 0.05,
            'overlap_range': (0.0, 1.0),
            'min_points': 512,
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [{
    'op': {
        'class': Clamp,
        'args': {
            'max_points': 4096,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}],
                },
            },
            'overlap': 0.5,
        },
    },
    'train_dataloader': {
        'class': GeoTransformerDataloader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'num_stages': 4,
            'voxel_size': 0.1,
            'search_radius': 0.25,
        },
    },
    'criterion': {
        'class': GeoTransformerCriterion,
        'args': {
            'loss': {
                'weight_coarse_loss': 1.0,
                'weight_fine_loss': 1.0,
            },
            'coarse_loss': {
                'positive_margin': 0.1,
                'negative_margin': 1.4,
                'positive_optimal': 0.1,
                'negative_optimal': 1.4,
                'log_scale': 24,
                'positive_overlap': 0.1,
            },
            'fine_loss': {
                'positive_radius': 0.05,
            },
        },
    },
    'val_dataset': {
        'class': BiTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'cache_filepath': './data/datasets/soft_links/ivision-pcr-data/../bi_temporal_pcr_cache.json',
            'split': 'val',
            'dataset_size': 1000,
            'rotation_mag': 45.0,
            'translation_mag': 0.5,
            'matching_radius': 0.05,
            'overlap_range': (0.0, 1.0),
            'min_points': 512,
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [{
    'op': {
        'class': Clamp,
        'args': {
            'max_points': 4096,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}],
                },
            },
            'overlap': 0.5,
        },
    },
    'val_dataloader': {
        'class': GeoTransformerDataloader,
        'args': {
            'batch_size': 1,
            'num_workers': 4,
            'num_stages': 4,
            'voxel_size': 0.1,
            'search_radius': 0.25,
        },
    },
    'test_dataset': None,
    'test_dataloader': None,
    'metric': {
        'class': GeoTransformerMetric,
        'args': {
            'acceptance_overlap': 0.0,
            'acceptance_radius': 0.1,
            'inlier_ratio_threshold': 0.05,
            'rmse_threshold': 0.2,
            'rre_threshold': 15.0,
            'rte_threshold': 0.3,
        },
    },
    'model': {
        'class': GeoTransformer,
        'args': {
            'model': {
                'ground_truth_matching_radius': 0.05,
                'num_points_in_patch': 32,
                'num_sinkhorn_iterations': 100,
            },
            'backbone': {
                'num_stages': 4,
                'init_voxel_size': 0.025,
                'kernel_size': 15,
                'base_radius': 2.5,
                'base_sigma': 2.0,
                'init_radius': 0.0625,
                'init_sigma': 0.05,
                'group_norm': 32,
                'input_dim': 1,
                'init_dim': 64,
                'output_dim': 256,
            },
            'geotransformer': {
                'input_dim': 1024,
                'hidden_dim': 256,
                'output_dim': 256,
                'num_heads': 4,
                'blocks': ['self', 'cross', 'self', 'cross', 'self', 'cross'],
                'sigma_d': 0.2,
                'sigma_a': 15,
                'angle_k': 3,
                'reduction_a': 'max',
            },
            'coarse_matching': {
                'num_targets': 128,
                'overlap_threshold': 0.1,
                'num_correspondences': 256,
                'dual_normalization': True,
            },
            'fine_matching': {
                'topk': 3,
                'acceptance_radius': 0.1,
                'mutual': True,
                'confidence_threshold': 0.05,
                'use_dustbin': False,
                'use_global_score': False,
                'correspondence_threshold': 3,
                'correspondence_limit': None,
                'num_refinement_steps': 5,
            },
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
