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
from configs.common.datasets.multi_task_learning.multi_mnist import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.multi_mnist.multi_mnist_lenet5 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.rgw import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 17037998
config['train_seeds'] = [90346228, 57169031, 1775258, 65554864, 27425410, 73247895, 511230, 70949959, 17025613, 46020511, 4728010, 47017320, 36597786, 14186586, 88546161, 86720198, 49768353, 20133637, 40068147, 18687689, 519776, 78851513, 92674475, 14065670, 57266381, 43602984, 92039348, 11797478, 61669345, 79781997, 55156373, 55937888, 47873908, 42519815, 53027063, 83301995, 76116202, 64112623, 43587642, 54930428, 78305732, 98812097, 24823974, 79468671, 87418273, 48826832, 75385652, 28861370, 67554461, 58147303, 37677983, 46776960, 39134112, 75181719, 83321378, 58488940, 4963992, 18650224, 72747271, 43859098, 44434806, 27496301, 89742335, 43414279, 32634178, 55415045, 81057619, 49965439, 9255000, 79586532, 83699793, 65165060, 61769277, 31586071, 80654754, 33080826, 48616289, 76249206, 12004606, 44995325, 63595757, 4375418, 97757782, 68949894, 42163128, 21267797, 44418706, 50843933, 92577829, 23668137, 44057226, 83048912, 45535300, 14072957, 56481502, 29634860, 57224629, 37188243, 76464606, 99733313]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/multi_mnist/multi_mnist_lenet5/rgw_run_1"
