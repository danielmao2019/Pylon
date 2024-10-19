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
from configs.common.optimizers.rgw import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 76003124
config['train_seeds'] = [98563465, 39918621, 58122191, 34995934, 69043406, 54390213, 1006257, 28141341, 42727128, 52594735, 37601490, 87138723, 7181494, 91580615, 53677777, 95036704, 18904102, 96948514, 27146949, 53213557, 6117978, 73961470, 72750248, 48280567, 32778775, 15028977, 21867767, 28755515, 36002704, 79720493, 10409177, 69903515, 66699296, 68521052, 16894720, 82052458, 67103506, 66222157, 57596507, 89075255, 80015472, 40145131, 56684354, 28778221, 32766190, 83279638, 80105615, 11407348, 85011742, 66717245, 75360485, 52832407, 90204037, 65710833, 31772373, 59000624, 74131854, 18622085, 5064869, 81458227, 76154424, 36344181, 53337484, 99409234, 93971438, 33298024, 58977442, 95372838, 72151792, 35275981, 40949115, 79434486, 31013959, 81756425, 98855219, 18497169, 7213523, 87534349, 71216309, 84637100, 73316786, 33673388, 96885671, 46793576, 35975962, 11627921, 70849803, 65692598, 5398839, 11974226, 84043895, 50645386, 13636347, 31356477, 70769479, 49732720, 62831477, 14428319, 51328493, 68780156]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/rgw_run_2"
