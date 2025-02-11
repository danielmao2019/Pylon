# This file is automatically generated by `./configs/benchmarks/change_detection/gen_bi_temporal.py`.
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
                'class': torch.optim.SGD,
                'args': {
                    'lr': 1.0e-03,
                    'momentum': 0.9,
                    'weight_decay': 1.0e-04,
                },
            },
        },
    },
    # scheduler config
    'scheduler': {
        'class': torch.optim.lr_scheduler.PolynomialLR,
        'args': {
            'optimizer': None,
            'total_iters': None,
            'power': 0.9,
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.change_detection.train.air_change import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.air_change import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.change_former import change_former_v4_config as model_config
config['model'] = model_config

# seeds
config['init_seed'] = 59183827
config['train_seeds'] = [62722339, 67284959, 84154863, 66520278, 92966304, 16854277, 20605639, 15978234, 68642997, 20147554, 80903760, 3298932, 5691899, 94986385, 44803408, 14109276, 91770754, 94225848, 48343630, 65781861, 18901695, 36058052, 84374349, 87221739, 94892946, 50933771, 54304633, 11587180, 13954011, 38629750, 90610975, 61840434, 11869734, 89375985, 90764120, 30089565, 83589, 90613154, 42428953, 66701624, 65646330, 40412914, 8530052, 32526419, 59646874, 13549512, 93723879, 14264051, 43730698, 80193565, 30738626, 83064655, 3935494, 40205399, 58485033, 39674036, 37544278, 80916523, 53492387, 53051698, 6096697, 98870847, 72846264, 35026695, 61673688, 30305111, 64070058, 66697630, 90304786, 86714226, 99238615, 15091426, 10145194, 76440134, 41067453, 18474051, 92312110, 80860584, 1481939, 85783794, 84007931, 31055728, 98860309, 13046919, 57151415, 54668949, 16649019, 51308089, 44609352, 99766816, 280953, 63980181, 18816237, 74698166, 34708225, 73981727, 86960031, 34976609, 88297261, 11856208]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/air_change/ChangeFormerV4_run_2"
