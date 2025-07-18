# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from criteria.vision_3d.point_cloud_registration.overlappredator_criterion.overlappredator_criterion import OverlapPredatorCriterion
from data.dataloaders.overlappredator_dataloader import OverlapPredatorDataloader
from data.datasets.pcr_datasets.bi_temporal_pcr_dataset import BiTemporalPCRDataset
from data.transforms.compose import Compose
from data.transforms.vision_3d.clamp import Clamp
from metrics.vision_3d.point_cloud_registration.overlappredator_metric.overlappredator_metric import OverlapPredatorMetric
from models.point_cloud_registration.overlappredator.overlappredator import OverlapPredator
from optimizers.single_task_optimizer import SingleTaskOptimizer
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer


config = {
    'runner': SupervisedSingleTaskTrainer,
    'work_dir': './logs/benchmarks/point_cloud_registration/bi_temporal_pcr_1.0/OverlapPredator_run_2',
    'epochs': 100,
    'init_seed': 90463292,
    'train_seeds': [77360698, 23224621, 37348686, 22401338, 42422221, 11697573, 28631718, 6523761, 42058742, 38697765, 51726996, 10607028, 23859812, 76750328, 83267333, 80268123, 10037633, 16332238, 36248845, 72205312, 62420036, 10716796, 35867171, 91199177, 16094125, 32886887, 69468720, 78380272, 78569799, 72823932, 49214234, 61545128, 76598070, 29202254, 11974520, 24026565, 4202861, 38318584, 48569364, 86608652, 91413930, 50309926, 40328456, 67025889, 26257835, 50170519, 36143387, 67274315, 83212640, 48971479, 66866256, 11957907, 17311184, 18917318, 62923134, 66853843, 63415944, 84370408, 13138931, 88739651, 68160870, 3831586, 3865588, 6099194, 10178528, 91974965, 65945595, 65793039, 60514449, 74512112, 84958326, 55418621, 84729824, 63285740, 48466072, 14622383, 24574864, 37884356, 35151387, 37468399, 41417532, 95804917, 14488323, 67490945, 15712280, 4376104, 89503576, 19216143, 63846276, 39259690, 92606014, 97014837, 57255053, 54502406, 28069597, 9289807, 95849475, 28675035, 33574251, 51973576],
    'val_seeds': [15897408, 52193391, 62960784, 85517521, 54306104, 99655658, 54754825, 87377269, 28159521, 82163142, 66392921, 69790430, 92063517, 90459412, 73538421, 90796979, 49033615, 78928513, 71995228, 13279219, 49823233, 40584857, 3681720, 69884644, 64281048, 32126712, 55010688, 60874822, 2424774, 1376522, 381822, 67209815, 2088643, 59793286, 55987127, 47456226, 64121386, 23336851, 45246407, 43445843, 96887469, 15273310, 26443650, 90542159, 60151748, 59650763, 26846823, 52581319, 15714496, 98954374, 39162210, 15022118, 54895734, 32639487, 44535018, 81224880, 42180742, 86696491, 87197674, 6902605, 39516272, 47629692, 61304381, 74979099, 19127886, 38553545, 35635394, 93547353, 80018894, 24519518, 11652194, 59972754, 71694420, 40373360, 98421179, 3688809, 1632701, 81867827, 15163657, 14317706, 4689572, 72216809, 67902173, 74026498, 26377025, 20305644, 95260846, 52196221, 11731705, 73651832, 23179395, 44858846, 60616091, 98244029, 4550416, 94783270, 52925074, 21652437, 46254524, 50318109],
    'test_seed': 85053414,
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
            'max_points': 8192,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}],
                },
            },
            'overlap': 1.0,
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
            'max_points': 8192,
        },
    },
    'input_names': [('inputs', 'src_pc'), ('inputs', 'tgt_pc')],
}],
                },
            },
            'overlap': 1.0,
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
