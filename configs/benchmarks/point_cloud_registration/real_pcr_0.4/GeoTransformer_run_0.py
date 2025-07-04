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
    'work_dir': './logs/benchmarks/point_cloud_registration/real_pcr_0.4/GeoTransformer_run_0',
    'epochs': 100,
    'init_seed': 78456540,
    'train_seeds': [63237206, 63873272, 42768870, 26725235, 50069626, 13833448, 88502737, 48077283, 80106479, 63701025, 13939593, 28100950, 81602880, 54189456, 3213077, 8371204, 92093229, 32441265, 90172628, 56340338, 26344445, 84237548, 39813335, 82305592, 64329255, 44568221, 70447499, 1160491, 72203856, 66363, 26132054, 78299707, 10987425, 52517973, 48846801, 90345113, 66744716, 27669584, 24913197, 13452765, 99935531, 77151517, 21984349, 18616064, 89170935, 42343762, 48968343, 48887030, 75687229, 88312356, 39772195, 29051550, 38153834, 39789970, 8444268, 81092151, 41409042, 847757, 37341195, 61617851, 78257616, 59136904, 60809483, 98107827, 57086590, 35501520, 57105354, 27277378, 92558185, 26052491, 77673572, 66324987, 62380458, 52026665, 2917022, 89866130, 37999958, 42814194, 66063719, 53947156, 58821009, 64163020, 80258153, 32873351, 83763787, 39609443, 9971770, 45530471, 87170069, 96098448, 49118962, 8904458, 56339503, 17895803, 3142987, 76057221, 89233737, 98343440, 30352424, 39064819],
    'val_seeds': [39504898, 34007634, 97057784, 14175945, 13994198, 24929686, 26078962, 20844040, 98956474, 12302317, 21632515, 73270241, 72238976, 81144428, 31804281, 58949424, 27862020, 37749582, 3640779, 31413369, 17920610, 83863331, 76582545, 89794634, 32059887, 61821755, 14348061, 94200922, 74651708, 39862434, 90993796, 69127986, 76161962, 45597725, 21079690, 31120052, 35807788, 70799004, 87850836, 48410032, 38179664, 53927800, 18944029, 6351732, 92814406, 49865843, 80793469, 70806924, 82567672, 14725372, 41434526, 58720282, 4914352, 24374810, 41597452, 36916277, 88086447, 25189564, 16884284, 21205519, 25569510, 16219524, 94165079, 99696044, 57362634, 83774059, 16186875, 61388222, 23358997, 63996332, 37418178, 12184794, 96607293, 65100598, 4945022, 65450021, 63928681, 87807616, 63668322, 24200245, 72959253, 95460421, 32116118, 95090808, 45195515, 64322959, 48073533, 5183442, 61402960, 25482498, 48148727, 93160479, 45080537, 73580566, 46043479, 78125412, 19172787, 38314281, 45011397, 21846659],
    'test_seed': 75524371,
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
            'overlap': 0.4,
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
            'overlap': 0.4,
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
