# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from criteria.vision_3d.point_cloud_registration.geotransformer_criterion.geotransformer_criterion import GeoTransformerCriterion
from data.dataloaders.geotransformer_dataloader import GeoTransformerDataloader
from data.datasets.pcr_datasets.synth_pcr_dataset import SynthPCRDataset
from data.transforms.compose import Compose
from data.transforms.vision_3d.random_rigid_transform import RandomRigidTransform
from metrics.vision_3d.point_cloud_registration.geotransformer_metric.geotransformer_metric import GeoTransformerMetric
from models.point_cloud_registration.geotransformer.geotransformer import GeoTransformer
from optimizers.single_task_optimizer import SingleTaskOptimizer
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer


config = {
    'runner': SupervisedSingleTaskTrainer,
    'work_dir': './logs/benchmarks/point_cloud_registration/synth_pcr_0.5/GeoTransformer_run_1',
    'epochs': 100,
    'init_seed': 32611073,
    'train_seeds': [48103225, 63555337, 22931043, 61616609, 49341628, 98011890, 38053985, 189863, 9142055, 97158482, 61723023, 91413598, 28940822, 46406028, 39841057, 75500318, 72375211, 5938914, 9030727, 95774988, 95635915, 87791125, 81192810, 96835151, 12843904, 90629446, 24028092, 43180687, 94035220, 66598449, 41452650, 59688438, 65856741, 46056302, 35650707, 28819290, 3289557, 12648464, 77289331, 78286045, 56469969, 900596, 29738662, 39070606, 48055661, 76166465, 84222508, 10542968, 10630044, 91666050, 78412921, 38297742, 92989983, 209940, 65865556, 9352016, 91091233, 23487379, 3464376, 47774816, 55814219, 35886129, 69577970, 56093226, 68954034, 22356855, 23761258, 76447665, 376502, 42612205, 38554598, 65426340, 73033663, 78388511, 89886888, 93853238, 15026872, 83133400, 59163641, 64641973, 9903663, 94024809, 49057749, 20689690, 33716318, 30096124, 98562561, 49553489, 90546816, 72552042, 34612442, 62836618, 17006575, 91170582, 70527933, 49772814, 75025150, 31956092, 42790075, 39959825],
    'val_seeds': [44680627, 44937869, 87675815, 22812761, 64316320, 48511758, 75692465, 12557115, 80929953, 43915947, 79617829, 66118217, 32492003, 89938625, 74493734, 52175849, 78138265, 20642294, 81235683, 79358633, 3201813, 11893136, 51776554, 93703942, 18460656, 19351922, 41223004, 7472911, 20554645, 94730591, 31186272, 88522387, 74672959, 52374096, 82392361, 48750998, 59622399, 13399767, 98033258, 20908165, 4880435, 77447603, 53477456, 5150970, 63913934, 31920324, 53890563, 62416022, 73149914, 99680835, 98789700, 7928729, 65893024, 36878982, 94412866, 31953435, 13775808, 81915237, 46309929, 12185346, 44284642, 20254446, 50645594, 77722451, 31325218, 60168905, 80625297, 23124267, 88509999, 80892067, 43027125, 53999371, 38029881, 48495188, 92885207, 87653863, 75780391, 37419438, 74082302, 54849906, 60024474, 48988219, 9948762, 63489227, 67986324, 77770052, 85115153, 80831776, 67864851, 4603699, 34514268, 45767710, 83117112, 9881973, 26570700, 62538544, 77301305, 9443958, 93675903, 81217175],
    'test_seed': 68287389,
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
