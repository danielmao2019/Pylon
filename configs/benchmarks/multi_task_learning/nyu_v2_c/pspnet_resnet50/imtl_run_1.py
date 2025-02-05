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
from configs.common.optimizers.imtl import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 85347666
config['train_seeds'] = [80192920, 86848872, 12172389, 57384766, 9410, 59869238, 41422042, 59878000, 40151450, 72527402, 53515236, 97081795, 63307429, 42051783, 56577126, 11081352, 28633685, 37349343, 18834537, 36546286, 22152094, 60299978, 11963486, 35116674, 77167152, 5630621, 22343568, 44395204, 12286841, 45792793, 58359659, 40969073, 91192550, 32779883, 65158488, 43661811, 90908517, 86348743, 24217427, 68218590, 56998009, 15960616, 64794046, 75273778, 60771827, 32517091, 91326333, 36402018, 39056297, 69237566, 3564057, 41807820, 24949921, 56168268, 2163944, 76769410, 14605469, 19414075, 15448481, 79118189, 92973975, 7475588, 52070339, 14656174, 66221320, 47091937, 60958910, 71856323, 2860733, 2046387, 32927976, 90644008, 80341964, 29687901, 60191003, 77537909, 45639364, 98713983, 31758396, 67738100, 55636022, 55934, 5333935, 86945761, 5987007, 4006183, 82220516, 48824347, 86030291, 61462124, 25257649, 81201025, 82368839, 226230, 65709018, 3524952, 59717763, 8941192, 49203015, 89021930]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/imtl_run_1"
