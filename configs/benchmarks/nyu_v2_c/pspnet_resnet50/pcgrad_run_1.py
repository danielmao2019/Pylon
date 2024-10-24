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
from configs.common.datasets.nyu_v2_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.pcgrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 22247090
config['train_seeds'] = [53760618, 34985036, 29815568, 2184216, 96728345, 2049401, 82724839, 1238013, 38362103, 25128325, 52694917, 19652686, 72708949, 52673411, 25410509, 84243987, 76373950, 87831483, 67814929, 93193810, 83747036, 98424706, 9714279, 93051523, 13508402, 79171164, 70844983, 49724114, 32443389, 70175517, 82201554, 29963167, 80271912, 75279898, 56957386, 47466381, 6338289, 52823824, 50181042, 21582370, 22838246, 29474683, 73181947, 96856237, 80939592, 79348123, 56313871, 91413992, 53039943, 54894886, 38469702, 43947248, 83525538, 77356845, 97696166, 81258928, 79598987, 77743700, 17996644, 85433986, 65688660, 68227603, 37252231, 15986817, 94508665, 95242602, 31945751, 73874550, 1667474, 22664564, 48137596, 19717539, 31340327, 80019210, 13919157, 74213714, 85484369, 30637830, 85562311, 56802706, 40761627, 51140960, 57254811, 64666808, 42816252, 68074021, 78551948, 11865003, 44293002, 68083206, 21789564, 55581628, 67780145, 54624066, 61349798, 15117910, 94387819, 85028027, 4648949, 46628826]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/pcgrad_run_1"
