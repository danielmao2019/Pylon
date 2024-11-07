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
from configs.common.datasets.multi_mnist import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_mnist.multi_mnist_lenet5 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.imtl import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 90628119
config['train_seeds'] = [43145825, 64293799, 74218702, 32194830, 91304037, 42586453, 70332076, 2973186, 60266223, 8897789, 80758430, 57530495, 50062039, 96261301, 89259280, 85079626, 35490186, 35433119, 97235762, 59379390, 44068422, 12171435, 31766417, 46079332, 87863855, 63962221, 65139091, 35165742, 66376935, 33072069, 62923518, 90409820, 83242491, 52444844, 45587269, 27282003, 81517896, 98368099, 15799930, 51077933, 24154293, 21162018, 61326647, 26340772, 42056400, 87252688, 45330738, 36702635, 93076884, 49975559, 70075310, 6056224, 12191487, 38426998, 82346099, 59311067, 80656424, 14307967, 73868070, 91297636, 33445159, 31850949, 2294423, 78820190, 17749726, 57502284, 25347729, 99267465, 6891926, 82144802, 30262454, 38526863, 60520307, 84570299, 38251790, 58825224, 84323963, 52877193, 67294, 44016617, 39154762, 34643981, 33405733, 86783491, 77611063, 21880799, 75107213, 6349082, 90679277, 7511195, 99853921, 24419906, 90270899, 20845584, 97590130, 64340442, 77564654, 51956763, 73468726, 68501812]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/imtl_run_0"
