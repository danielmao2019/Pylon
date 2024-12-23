# This file is automatically generated by `./configs/benchmarks/multi_task_learning.py`.
# Please do not attempt to modify manually.
import torch
import schedulers


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
    'val_dataset': None,
    'val_dataloader': None,
    'test_dataset': None,
    'test_dataloader': None,
    'criterion': None,
    'metric': None,
    # model config
    'model': None,
    # optimizer config
    'optimizer': None,
    'scheduler': {
        'class': torch.optim.lr_scheduler.LambdaLR,
        'args': {
            'lr_lambda': {
                'class': schedulers.WarmupLambda,
                'args': {},
            },
        },
    },
}

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.imtl import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 33704807
config['train_seeds'] = [33910780, 9343458, 83816647, 78210551, 70619969, 37616216, 57384541, 4560884, 63220284, 53083032, 64971825, 32893303, 493138, 97565504, 68887882, 38789784, 28445145, 20220229, 27506516, 55737781, 73032777, 33100686, 97875393, 20561946, 58653025, 96907992, 82272037, 37532291, 40999933, 31732395, 26632477, 91696878, 63585747, 21683021, 38928160, 32770931, 48187956, 99654912, 47137409, 63069780, 34580249, 77781736, 69111198, 57212227, 49723446, 89473998, 31335883, 43344207, 42339559, 45982406, 4859006, 48478996, 16554597, 77312675, 24971374, 31953849, 98422419, 90446735, 76389292, 25550226, 12863859, 87366773, 44397547, 54440831, 39627498, 38407829, 95420872, 82177741, 89959662, 50698812, 25059078, 11249244, 85080160, 44042971, 32888359, 61800463, 46760105, 8339462, 66012688, 53781405, 69011024, 36552459, 31183220, 88140650, 24603822, 12826666, 49753993, 27812706, 60691363, 68624758, 5192398, 10315715, 18361667, 93951254, 3774994, 87971546, 5849904, 50221676, 2775709, 26385303]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/imtl_run_2"
