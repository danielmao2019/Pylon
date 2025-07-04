# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from criteria.vision_3d.point_cloud_registration.geotransformer_criterion.geotransformer_criterion import GeoTransformerCriterion
from data.dataloaders.geotransformer_dataloader import GeoTransformerDataloader
from data.datasets.pcr_datasets.real_pcr_dataset import RealPCRDataset
from data.transforms.compose import Compose
from data.transforms.vision_3d.pcr_translation import PCRTranslation
from metrics.vision_3d.point_cloud_registration.geotransformer_metric.geotransformer_metric import GeoTransformerMetric
from models.point_cloud_registration.geotransformer.geotransformer import GeoTransformer
from optimizers.single_task_optimizer import SingleTaskOptimizer
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer


config = {
    'runner': SupervisedSingleTaskTrainer,
    'work_dir': './logs/benchmarks/point_cloud_registration/real_pcr_0.5/GeoTransformer_run_2',
    'epochs': 100,
    'init_seed': 18250117,
    'train_seeds': [36065661, 83994145, 22177182, 97636575, 6600893, 25256834, 88160551, 32971409, 6170598, 16057766, 43140656, 43522812, 9478223, 73475725, 94850068, 10558901, 49056875, 80394633, 66397453, 55203237, 61117287, 97735024, 19500588, 76381290, 45190796, 31828909, 35576195, 12328713, 86163743, 48968728, 21220433, 13759145, 29673585, 33461172, 24965425, 62463345, 88802112, 90996601, 30679642, 98787406, 36898027, 41806734, 55512412, 39600619, 72496012, 83191664, 45931718, 73428143, 82472015, 6057345, 93451565, 85133042, 43976447, 90122870, 27415985, 29681080, 31440430, 27204213, 30672127, 89372548, 7686079, 63686423, 87902901, 58080792, 71578588, 35022439, 6341716, 34229406, 1139105, 70626484, 62133388, 76291384, 20271496, 35523510, 54249197, 75258033, 71496990, 43902338, 27015621, 35677652, 95829635, 47925275, 11047736, 21142854, 45595428, 73776569, 61184128, 23493237, 17064001, 23632958, 76760291, 27994429, 5802664, 76543782, 89209677, 42957496, 5723808, 91251176, 65571161, 57451429],
    'val_seeds': [98262866, 36661610, 77461331, 62175685, 56026280, 59486594, 35848065, 98561697, 70730564, 9856000, 85933874, 5199541, 82798173, 99837376, 2352645, 10759996, 35857742, 88192848, 73541583, 64082215, 91529246, 27123619, 19582729, 52690502, 27383051, 77498853, 7235717, 77612045, 68207437, 76063268, 33637485, 92257399, 93300742, 87121521, 34444725, 70117276, 60886533, 88793528, 57689369, 97411716, 58885060, 7210828, 14400320, 55396395, 61101738, 81743779, 6183148, 60618621, 81493556, 77918594, 3666148, 48911994, 74719212, 74272916, 82950955, 89757435, 48406986, 33515441, 34447871, 57846402, 84674734, 41168273, 18558920, 45630260, 15528706, 55753661, 12110711, 98250654, 15627935, 1585886, 99396532, 58756753, 27160337, 25375379, 30599822, 1906702, 2962798, 30040888, 19069071, 22131606, 47459552, 40966220, 41998878, 81358179, 21087917, 64030627, 47093404, 54858975, 32300617, 40863632, 10670501, 53161229, 29456041, 63520238, 81606193, 48797627, 86430576, 78536978, 77563520, 61276624],
    'test_seed': 7840945,
    'train_dataset': {
        'class': RealPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'cache_dirname': 'real_pcr_cache',
            'split': 'train',
            'voxel_size': 10.0,
            'min_points': 512,
            'max_points': 4096,
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [(
    {
            'class': PCRTranslation,
            'args': {},
        },
    [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')]
)],
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
        'class': RealPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'cache_dirname': 'real_pcr_cache',
            'split': 'val',
            'voxel_size': 10.0,
            'min_points': 512,
            'max_points': 4096,
            'transforms_cfg': {
                'class': Compose,
                'args': {
                    'transforms': [(
    {
            'class': PCRTranslation,
            'args': {},
        },
    [('inputs', 'src_pc'), ('inputs', 'tgt_pc'), ('labels', 'transform')]
)],
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
