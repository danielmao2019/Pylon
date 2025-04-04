# This file is automatically generated by `./configs/benchmarks/multi_task_learning/gen_multi_task_learning.py`.
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
from configs.common.datasets.multi_task_learning.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.alignedmtl import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 11924975
config['train_seeds'] = [48737565, 15057048, 11570690, 76082265, 92884135, 28773038, 57853661, 55007135, 39120935, 85023712, 71793826, 14326558, 8398679, 5996044, 32901492, 35618262, 424452, 76247952, 20428179, 83514128, 74561142, 58435962, 57116830, 52865391, 41095253, 49872932, 18535807, 7055492, 20015291, 84984465, 59654764, 34585784, 74434302, 68076165, 82626320, 96369840, 87258715, 22496950, 32378442, 69578425, 87515055, 6398013, 60605905, 66527927, 75778886, 29676889, 69395802, 24115599, 76036279, 98225029, 98445106, 49591337, 17545636, 99408442, 1958461, 37808787, 35203734, 53217135, 35553538, 45730634, 16852111, 56259710, 9428430, 28473984, 54097949, 91863292, 77044568, 65743685, 2384210, 54700894, 72840953, 12145048, 55956795, 50139451, 88491797, 80942124, 31859404, 22663289, 43843396, 17762807, 73256806, 51074693, 36608927, 62336400, 70687803, 31077860, 70926737, 23682843, 89867691, 56168607, 43051732, 90423391, 68855465, 32541617, 39061123, 56985616, 88699009, 84845804, 70279888, 16147399]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/celeb_a/celeb_a_resnet18/alignedmtl_run_2"
