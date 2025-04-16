# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
import torch
import optimizers


config = {
    'runner': None,
    'work_dir': None,
    'epochs': 100,
    # seeds
    'init_seed': None,
    'train_seeds': None,
    # dataset config
    'train_dataset': None,
    'train_dataloader': None,
    'criterion': None,
    'val_dataset': None,
    'val_dataloader': None,
    'test_dataset': None,
    'test_dataloader': None,
    'metric': None,
    # model config
    'model': None,
    # optimizer config
    'optimizer': {
        'class': optimizers.SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': torch.optim.Adam,
                'args': {
                    'params': None,
                    'lr': 1.0e-4,
                    'weight_decay': 1.0e-06,
                },
            },
        },
    },
    # scheduler config
    'scheduler': {
        'class': torch.optim.lr_scheduler.StepLR,
        'args': {
            'optimizer': None,
            'step_size': 1000,
            'gamma': 0.95,
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# data config
from configs.common.datasets.point_cloud_registration.train.overlappredator_real_pcr_data_cfg import data_cfg as train_data_cfg
train_data_cfg['train_dataset']['args']['overlap'] = 1.0
config.update(train_data_cfg)
from configs.common.datasets.point_cloud_registration.val.overlappredator_real_pcr_data_cfg import data_cfg as val_data_cfg
val_data_cfg['val_dataset']['args']['overlap'] = 1.0
config.update(val_data_cfg)

# model config
from configs.common.models.point_cloud_registration.overlappredator_cfg import model_cfg
config['model'] = model_cfg

from configs.common.criteria.point_cloud_registration.overlappredator_criterion_cfg import criterion_cfg
config['criterion'] = criterion_cfg

from configs.common.metrics.point_cloud_registration.overlappredator_metric_cfg import metric_cfg
config['metric'] = metric_cfg

# seeds
config['init_seed'] = 97414384
config['train_seeds'] = [75989716, 27235948, 75305353, 83047424, 70624366, 2612669, 21338494, 71193421, 93945913, 32559032, 37430733, 38769408, 70008762, 47702228, 78184013, 14986642, 51310281, 88105816, 65626695, 78457156, 51843911, 20489012, 90290598, 77974537, 84670786, 61059227, 77173412, 21335549, 64216119, 79795696, 69793906, 8398168, 93295655, 37986491, 87434468, 82542723, 60036357, 20836893, 67609061, 78229919, 13976541, 61321729, 77928593, 5343362, 50216570, 49943817, 12674584, 76684956, 32595650, 78656198, 22900673, 91370504, 99544694, 92228123, 59988300, 1017896, 45067951, 87812583, 89015863, 84235236, 90812641, 64284740, 63728961, 17116520, 20802444, 72754922, 83566329, 79003199, 23264110, 19674755, 19053990, 76272316, 6959740, 86784994, 37892269, 53638769, 72481180, 29449101, 10554695, 3922025, 81083741, 49390503, 49338987, 97180594, 17422268, 55616880, 18932449, 92247044, 12797767, 86831231, 94702832, 85393763, 41858123, 1100671, 14981646, 82829107, 28754396, 21367486, 70588519, 35530489]

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/real_pcr/overlap_1.0/OverlapPredator_run_1"
