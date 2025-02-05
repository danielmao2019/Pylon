# This file is automatically generated by `./configs/benchmarks/change_detection/gen_gan.py`.
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

from runners.gan_trainers import GANTrainer
config['runner'] = GANTrainer

# dataset config
from configs.common.datasets.gans.mnist import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.gans.gan import model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.gans import gan_optimizer_config as optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 2637337
config['train_seeds'] = [97840199, 86338973, 32618950, 93889890, 39461557, 13302001, 21930100, 27629594, 24360617, 29339963, 2371435, 40520242, 28562553, 40151289, 95939297, 82427740, 62694149, 90996581, 53018937, 77439232, 2478901, 33857401, 72156421, 47469107, 48643653, 16794162, 36553739, 55940195, 83582287, 44936660, 27582579, 76740302, 61539630, 55055173, 15987861, 93004776, 39837163, 98583878, 31945606, 7026137, 88510124, 48449940, 83211931, 10432831, 69746971, 19784891, 69905779, 18781294, 87530055, 82106119, 47060049, 62830545, 59753404, 32564882, 38173213, 79881258, 70493106, 59034395, 33730548, 71677524, 49279954, 55047409, 68660906, 80697917, 49898608, 91510000, 22520411, 49894725, 73913298, 76289019, 88544603, 47228272, 20837204, 64217085, 23037053, 68831718, 70668757, 13402882, 37696208, 16613563, 6623913, 81860029, 7822141, 85142821, 34474196, 48064507, 39809946, 55200495, 86082098, 8707409, 65763143, 39387191, 30934712, 52224648, 9135685, 44267717, 48982428, 50489121, 37692249, 82011712]

# work dir
config['work_dir'] = "./logs/benchmarks/gans/gan/gan_run_0"
