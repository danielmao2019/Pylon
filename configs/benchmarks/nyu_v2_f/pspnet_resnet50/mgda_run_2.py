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
from configs.common.datasets.nyu_v2_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.mgda import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 81192046
config['train_seeds'] = [25051434, 47174078, 39926886, 17338454, 40106498, 49198738, 78935889, 50315115, 11060114, 87131876, 45039157, 59894081, 64728924, 4981737, 5784151, 95480837, 36164134, 58242172, 74805681, 56370787, 92937263, 48176637, 5926761, 88338316, 2662659, 11407088, 87277544, 38878842, 65604212, 99986599, 34686013, 99061415, 71691007, 93979096, 46118868, 15121371, 23433039, 72707580, 17906831, 31658504, 87223406, 37075864, 35698434, 39680743, 11916154, 82572522, 18058605, 68136548, 64276585, 14829432, 89451437, 66436094, 71734159, 34225227, 99948529, 23348268, 84599918, 74045571, 70183999, 45341739, 53070833, 97373754, 95424049, 38421772, 44421391, 45093054, 10014474, 68421021, 5904017, 55412242, 98900936, 23423410, 33944119, 87733214, 20170156, 17257870, 78314460, 77552546, 66055321, 98855087, 8213058, 61665155, 605586, 20673626, 56717735, 94752755, 78805057, 84769835, 1730140, 83546709, 3777461, 76646359, 60673940, 41122673, 19089021, 7739954, 52094324, 85603283, 16865686, 712127]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/mgda_run_2"
