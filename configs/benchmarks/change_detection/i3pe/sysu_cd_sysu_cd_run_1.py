# This file is automatically generated by `./configs/benchmarks/change_detection/gen_i3pe.py`.
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
                'class': schedulers.lr_lambdas.WarmupLambda,
                'args': {},
            },
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.change_detection.train.i3pe_sysu_cd import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.sysu_cd import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.i3pe_model import model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 19128160
config['train_seeds'] = [43473712, 20227639, 53609980, 57487025, 40529993, 35558183, 96837917, 98726945, 13406634, 97449195, 5250072, 55120211, 33373004, 32406570, 73970754, 51586494, 37008475, 20769675, 73667080, 75462102, 89254368, 35069660, 88004427, 27889147, 74983281, 62415307, 97358013, 11459247, 26050777, 41650093, 32078254, 58367892, 36821605, 91211417, 95068066, 13821482, 54495261, 41455438, 72859046, 50355566, 45717092, 16244414, 45044515, 70206378, 25577115, 53952342, 79118277, 15118218, 57289140, 66622195, 41863565, 14041170, 81753171, 85791439, 78272399, 42266226, 58384630, 33170414, 75873976, 60327673, 3555875, 19890909, 11294585, 73761896, 87949438, 858577, 36552179, 43437966, 41061525, 26270099, 87992948, 98502868, 16201834, 96874994, 85800806, 76797547, 63152489, 95682304, 32290144, 78950608, 35648105, 10490945, 5688017, 93527822, 82137586, 65229465, 95377042, 62587275, 17465897, 29016272, 66821422, 53664319, 69729820, 80734649, 41632044, 64934404, 61104571, 3803761, 45607100, 14794558]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/i3pe/sysu_cd_sysu_cd_run_1"
