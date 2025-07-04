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
    'work_dir': './logs/benchmarks/point_cloud_registration/synth_pcr_1.0/GeoTransformer_run_2',
    'epochs': 100,
    'init_seed': 18362205,
    'train_seeds': [40654666, 84300225, 72664933, 18480290, 96485444, 56862959, 90771905, 1383515, 79570714, 32929458, 4694727, 64743244, 88995347, 56775342, 85844457, 84142512, 81845068, 92068131, 1457828, 86881216, 64508716, 9446670, 96271122, 56260112, 48571301, 77021942, 72136042, 21135655, 94594686, 28877770, 83835509, 89737118, 56827529, 67803918, 16356065, 93046126, 39770542, 23434683, 71172847, 22945698, 70151023, 57381969, 57283646, 13217918, 28492477, 34920245, 63730294, 89231825, 69153440, 33949781, 60699288, 91404245, 51577747, 31344714, 81993402, 37320566, 62209867, 92191622, 47560987, 70056410, 89959611, 21383578, 48442125, 52126221, 18097586, 9329869, 23821763, 90991031, 44959409, 65788622, 97513349, 7766848, 68307110, 47433739, 50798605, 87255956, 59865443, 77492805, 63630579, 29922945, 31717298, 21670904, 93840038, 77424673, 97055891, 48542324, 3715777, 75731274, 83568607, 4136139, 25503444, 68149222, 94035241, 95157934, 60144988, 65985799, 94666624, 44608544, 2813061, 99891661],
    'val_seeds': [45994438, 96359006, 97016441, 11385671, 76577874, 70509369, 83863209, 52357961, 34385894, 33298994, 74610847, 29868385, 47821797, 30514147, 74118224, 12012552, 25486376, 64286406, 72941480, 6007567, 35387932, 11562324, 63711203, 29118794, 641621, 72227570, 20490863, 62927255, 56923771, 35300443, 46420944, 70544304, 81881343, 62959857, 63693731, 42460540, 12846746, 85544893, 29430213, 49189532, 91935101, 62870112, 98360025, 32171744, 17760493, 5489589, 95034583, 22235688, 59744767, 91094654, 66660421, 57628721, 60395703, 40126605, 18653240, 15364550, 95920516, 95935843, 46897239, 23477630, 12341092, 39458767, 67570886, 63356166, 89091405, 79617318, 30194404, 18665605, 31570755, 77393538, 91685471, 89208924, 41175186, 75155330, 22104489, 81410443, 30396023, 40286980, 32649087, 94393019, 893220, 87888054, 79998901, 78302371, 4719931, 72610936, 97772970, 91026876, 23177561, 66014526, 18738235, 20254216, 61087068, 16680858, 81215795, 18767590, 80421091, 71468695, 93646762, 48822697],
    'test_seed': 48228766,
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
            'overlap': 1.0,
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
            'overlap': 1.0,
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
