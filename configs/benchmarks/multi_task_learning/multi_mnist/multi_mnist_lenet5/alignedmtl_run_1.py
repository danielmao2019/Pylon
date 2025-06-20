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
    'val_seeds': None,
    'test_seed': None,
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

from runners import SupervisedMultiTaskTrainer
config['runner'] = SupervisedMultiTaskTrainer

# dataset config
from configs.common.datasets.multi_task_learning.multi_mnist import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.multi_mnist.multi_mnist_lenet5 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.alignedmtl import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 5306257
config['train_seeds'] = [32759715, 81121924, 44741918, 61404464, 61268063, 15464166, 46882106, 45819525, 70868020, 36458279, 22369003, 88758823, 40692166, 42527240, 40632929, 68397382, 77170792, 15905847, 2111014, 47040573, 18618311, 81710664, 87788704, 57337753, 17320960, 38722153, 86046076, 32969151, 92935620, 26033378, 45514867, 76170891, 2909869, 4297067, 41662701, 55942697, 47013664, 56327789, 32915613, 55315972, 20065560, 81334382, 56449816, 98545309, 98658016, 33588500, 39585133, 71488993, 73846709, 95533636, 18545893, 41758442, 91596601, 87388552, 69425015, 88269690, 87407553, 68368177, 57679088, 58875748, 28522338, 77893579, 89408919, 77796044, 93196157, 17611037, 66529145, 10940639, 28612734, 99322614, 92943118, 724221, 88516193, 23670024, 16831552, 46515078, 78490425, 3382682, 99479429, 36641631, 72987458, 24489528, 73802624, 92521215, 11306535, 37314829, 87728170, 48958006, 91027711, 97616817, 56000898, 14706263, 74941763, 91936386, 24870535, 23884891, 26908882, 32989170, 99116640, 20782428]
config['val_seeds'] = [23235008, 31780107, 522977, 3739979, 41677738, 16214327, 6873522, 64710595, 50909791, 78088679, 84139547, 40226361, 55700568, 75604221, 18110262, 78224744, 66760606, 23204497, 36875531, 69858453, 97024788, 99088189, 53084953, 201651, 20196966, 77380127, 96706187, 65857369, 16455418, 54942536, 60080868, 25821860, 12928512, 91036795, 85380027, 43635401, 29414646, 24101008, 44112111, 29359997, 42232960, 10280709, 27288465, 47325505, 71321027, 6270116, 26336018, 21276084, 86634914, 45164180, 33888236, 84402571, 34769938, 80575873, 73753403, 98438503, 66055930, 34049260, 36644371, 836063, 5332154, 7938476, 98711343, 49974794, 88631574, 70221557, 52953110, 44986951, 37098650, 11406029, 37706434, 31134936, 42652701, 15819701, 36830195, 91547378, 5782577, 8115558, 19671791, 55336205, 14312486, 1241110, 12295413, 65396954, 71229022, 32146598, 80314081, 77218902, 38292108, 97466489, 31137072, 55804251, 86251966, 17958162, 60789509, 24166288, 72131240, 62066813, 11211295, 25399117]
config['test_seed'] = 72257123

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/multi_mnist/multi_mnist_lenet5/alignedmtl_run_1"
