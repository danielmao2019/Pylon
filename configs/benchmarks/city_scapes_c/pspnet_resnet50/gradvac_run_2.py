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
from configs.common.datasets.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.gradvac import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 74088955
config['train_seeds'] = [17895685, 17572741, 13994997, 25399719, 21606145, 49753221, 60061304, 20277735, 36073026, 29970336, 2550005, 38427599, 58946857, 85914116, 58992151, 55888455, 23597754, 36499054, 94950491, 10376289, 39039248, 90028110, 27233041, 90598495, 21537242, 25825231, 40609448, 37266618, 9889365, 89685619, 35636317, 36872922, 11224861, 7493024, 43954839, 16180768, 68275778, 8209597, 30334843, 50937520, 50180771, 94229780, 42706453, 43911603, 7493123, 34200144, 3050146, 21912649, 72785407, 22852654, 67469242, 96669011, 2142408, 32126451, 80007861, 67215853, 35444452, 7154087, 33488362, 18488572, 5702925, 40114521, 82087333, 5551530, 94969949, 43926736, 62365189, 58569745, 9601906, 62479243, 3951208, 8404811, 60421847, 61973697, 80201767, 45724071, 3677206, 63325108, 53607945, 77331010, 53646076, 80437048, 11115830, 86615975, 55799705, 60490680, 16644310, 35683683, 89956596, 95190002, 54141327, 83766695, 85991121, 78411708, 60496524, 79702105, 86593307, 53092905, 44700883, 29145996]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/gradvac_run_2"
