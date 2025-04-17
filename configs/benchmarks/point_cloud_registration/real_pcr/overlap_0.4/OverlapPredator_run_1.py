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
train_data_cfg['train_dataset']['args']['overlap'] = 0.4
config.update(train_data_cfg)
from configs.common.datasets.point_cloud_registration.val.overlappredator_real_pcr_data_cfg import data_cfg as val_data_cfg
val_data_cfg['val_dataset']['args']['overlap'] = 0.4
config.update(val_data_cfg)

# model config
from configs.common.models.point_cloud_registration.overlappredator_cfg import model_cfg
config['model'] = model_cfg

from configs.common.criteria.point_cloud_registration.overlappredator_criterion_cfg import criterion_cfg
config['criterion'] = criterion_cfg

from configs.common.metrics.point_cloud_registration.overlappredator_metric_cfg import metric_cfg
config['metric'] = metric_cfg

# seeds
config['init_seed'] = 88989842
config['train_seeds'] = [7161178, 30505168, 88481842, 99407827, 48971676, 93706727, 81143164, 98408201, 66439044, 64028192, 69208609, 66031129, 22557917, 48931621, 23349905, 95219274, 6033025, 79300313, 40001007, 69610631, 90082915, 28252437, 36853129, 89567241, 65892900, 27747806, 26677163, 16501217, 12430133, 12807777, 66001962, 99852390, 84456041, 72559457, 46340236, 67952545, 78955308, 47129767, 90448226, 1233263, 39001210, 65058412, 31325452, 67277831, 71890723, 95561828, 122300, 67616664, 76005052, 30478722, 31346673, 76680821, 82574185, 96242711, 26552930, 6578275, 24690929, 98440795, 60404767, 395809, 71700690, 74934372, 54373414, 48307957, 68179135, 59179470, 29293230, 91751632, 90963539, 9926455, 14388447, 86797322, 15412448, 76830768, 29894258, 31332741, 760295, 24622889, 60180498, 71595492, 38789027, 45076167, 95040981, 667058, 20043952, 93381526, 24333210, 91198007, 35462174, 45817126, 3071792, 67097615, 37862989, 20611136, 82287700, 4555006, 83472029, 89660671, 81468719, 78050197]

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/real_pcr/overlap_0.4/OverlapPredator_run_1"
