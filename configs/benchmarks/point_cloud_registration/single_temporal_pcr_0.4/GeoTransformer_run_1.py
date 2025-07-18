# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from criteria.vision_3d.point_cloud_registration.geotransformer_criterion.geotransformer_criterion import GeoTransformerCriterion
from data.dataloaders.geotransformer_dataloader import GeoTransformerDataloader
from data.datasets.pcr_datasets.single_temporal_pcr_dataset import SingleTemporalPCRDataset
from data.transforms.compose import Compose
from data.transforms.vision_3d.clamp import Clamp
from metrics.vision_3d.point_cloud_registration.geotransformer_metric.geotransformer_metric import GeoTransformerMetric
from models.point_cloud_registration.geotransformer.geotransformer import GeoTransformer
from optimizers.single_task_optimizer import SingleTaskOptimizer
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer


config = {
    'runner': SupervisedSingleTaskTrainer,
    'work_dir': './logs/benchmarks/point_cloud_registration/single_temporal_pcr_0.4/GeoTransformer_run_1',
    'epochs': 100,
    'init_seed': 8422542,
    'train_seeds': [57161962, 3807498, 30442024, 21037812, 78100738, 24767300, 12051045, 39043679, 59808323, 89988100, 62428907, 97117956, 33759940, 19411298, 8890763, 37227111, 67885491, 79699803, 83371426, 11660029, 60741401, 39602079, 79183615, 68185580, 64704850, 65218903, 80196493, 74648085, 12155344, 95095203, 33121829, 29716626, 47602820, 84130420, 11508571, 46921993, 34955390, 32225288, 96549513, 95812150, 55034099, 63099088, 34619728, 44195343, 79931984, 77853055, 80530183, 91300241, 34339577, 93769052, 63596048, 8316200, 37767628, 40549355, 2084287, 66348482, 75999179, 51536329, 22115723, 21133273, 76570437, 7825144, 38093111, 61069524, 9006106, 24240273, 82335968, 70850810, 7423806, 85615768, 62800598, 80523442, 5985554, 88245924, 13629764, 46892571, 92525736, 65035478, 2050021, 85162735, 25818490, 81913634, 79604560, 93982298, 96276794, 85381495, 29252136, 24191658, 99328098, 65919510, 26766299, 80341144, 54317777, 27899179, 56765461, 6534785, 47278176, 39584506, 1300380, 8282196],
    'val_seeds': [98944585, 1340152, 44574267, 60790599, 29681563, 91218064, 97584264, 43922409, 50105363, 1121447, 14543130, 94786440, 18716892, 79471151, 5728755, 13327586, 96110382, 21925350, 23495336, 15224079, 17923188, 99262452, 66015487, 89393585, 30500794, 73047431, 24079438, 52672879, 3325617, 66606900, 9212303, 86837156, 77897933, 79756286, 70432232, 71813467, 10796072, 38713931, 66239989, 14608899, 54040285, 44956520, 3904216, 49155782, 90773595, 53671160, 91575999, 20396695, 52775665, 27764318, 7183290, 61042825, 61793095, 73474543, 74398520, 66918330, 14095213, 57308977, 92454242, 23306176, 40808199, 81444533, 76827147, 89359385, 84609551, 97892838, 95031515, 82000057, 87980166, 76938877, 60602315, 75018005, 25625065, 7716828, 95314097, 29860808, 48476422, 54286047, 5467574, 51483717, 56619500, 35026191, 43832589, 20121474, 44041171, 28511407, 79883314, 51710965, 67490306, 54397064, 95624616, 59024911, 8825739, 29811864, 81010295, 98446495, 49062732, 30652969, 46085096, 41368291],
    'test_seed': 62919936,
    'train_dataset': {
        'class': SingleTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'cache_filepath': './data/datasets/soft_links/ivision-pcr-data/../single_temporal_pcr_cache.json',
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
            'max_points': 8192,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}],
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
        'class': SingleTemporalPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'cache_filepath': './data/datasets/soft_links/ivision-pcr-data/../single_temporal_pcr_cache.json',
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
            'max_points': 8192,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}],
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
