# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from criteria.vision_3d.point_cloud_registration.overlappredator_criterion.overlappredator_criterion import OverlapPredatorCriterion
from data.dataloaders.overlappredator_dataloader import OverlapPredatorDataloader
from data.datasets.pcr_datasets.real_pcr_dataset import RealPCRDataset
from data.transforms.compose import Compose
from data.transforms.vision_3d.pcr_translation import PCRTranslation
from metrics.vision_3d.point_cloud_registration.overlappredator_metric.overlappredator_metric import OverlapPredatorMetric
from models.point_cloud_registration.overlappredator.overlappredator import OverlapPredator
from optimizers.single_task_optimizer import SingleTaskOptimizer
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer


config = {
    'runner': SupervisedSingleTaskTrainer,
    'work_dir': './logs/benchmarks/point_cloud_registration/real_pcr_0.5/OverlapPredator_run_1',
    'epochs': 100,
    'init_seed': 12521741,
    'train_seeds': [72777452, 80982450, 81245102, 98482718, 86994697, 7214744, 56667799, 54885856, 93179083, 68741978, 13219962, 49836972, 42631058, 80501127, 97203190, 57263409, 35441008, 68093731, 79679352, 94980118, 70045092, 71144186, 77211779, 55033200, 63706454, 27603747, 31879067, 23226016, 30325260, 99468195, 56274717, 65424218, 75774713, 39060625, 92621518, 41435608, 57826894, 40870347, 36668332, 94630625, 14389277, 34315768, 18403888, 18010622, 48622710, 12151704, 43106677, 3003139, 34249559, 95369849, 42601046, 75744121, 55130288, 42929503, 44380430, 63538690, 5355359, 92843519, 69815184, 59901483, 42489954, 56006848, 83211852, 90187692, 1763668, 29489, 56292060, 5888540, 2434854, 48549324, 97328156, 72382316, 26482090, 51603346, 30675535, 91495637, 68019069, 1881594, 93335893, 69986576, 63891552, 73441988, 37536422, 44035754, 56212361, 39081849, 32558452, 57669382, 67554634, 19954189, 3580263, 48825546, 38045028, 55232886, 40311149, 56025545, 82774271, 99515754, 97099590, 86570926],
    'val_seeds': [5869553, 20553740, 31211787, 6034104, 39012881, 60132095, 59560779, 42649024, 47066282, 41028294, 42161715, 68935270, 47096690, 63336558, 5504260, 35100109, 14083984, 22516377, 77091653, 94166196, 96544870, 49145281, 49475751, 30592363, 56114368, 90213012, 55769907, 11002465, 6747417, 79477054, 84209714, 68721608, 52521450, 84475973, 12520109, 41010429, 92052024, 18221326, 43683476, 63957359, 40399848, 31507570, 67810015, 38648887, 3210165, 62242846, 53520259, 81999784, 22895802, 22591698, 47986918, 22839461, 50656068, 95334769, 21424749, 54794154, 27973770, 65285902, 64347024, 77879338, 46496368, 40623389, 17939146, 16991323, 17790421, 84797520, 54087709, 1418522, 96855617, 87113902, 88548657, 23044131, 69634361, 73879840, 10926320, 6383539, 58947164, 18899090, 56493917, 77828089, 19660340, 26493827, 33874488, 41716336, 25954525, 28256922, 74427441, 77160031, 64404498, 43572912, 66108322, 47248907, 99526431, 94926526, 13984477, 56132880, 48558256, 86774917, 16312372, 23114852],
    'test_seed': 60658739,
    'train_dataset': {
        'class': RealPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'cache_dirname': 'real_pcr_cache',
            'split': 'train',
            'voxel_size': 10.0,
            'min_points': 512,
            'max_points': 8192,
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
        'class': RealPCRDataset,
        'args': {
            'data_root': './data/datasets/soft_links/ivision-pcr-data',
            'gt_transforms_filepath': './data/datasets/soft_links/ivision-pcr-data/gt_transforms.json',
            'cache_dirname': 'real_pcr_cache',
            'split': 'val',
            'voxel_size': 10.0,
            'min_points': 512,
            'max_points': 8192,
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
