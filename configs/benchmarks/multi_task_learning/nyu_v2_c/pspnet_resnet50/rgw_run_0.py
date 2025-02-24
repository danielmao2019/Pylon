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
from configs.common.datasets.multi_task_learning.nyu_v2_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.rgw import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 72952495
config['train_seeds'] = [64117929, 552396, 9583038, 64711426, 37663632, 77079761, 15366721, 83912022, 94221053, 48481423, 11262660, 15083705, 79271897, 9621062, 79109320, 38573234, 64563830, 93269919, 80688048, 51238303, 28782754, 34227070, 53251321, 18606259, 37019035, 5099468, 3091338, 43078915, 81248731, 5368433, 82200090, 71808456, 23574738, 89393568, 55916621, 96171620, 33853598, 64264076, 78122288, 74550494, 25719336, 62254985, 96914724, 20443259, 44815638, 3692570, 31639182, 17022429, 80445639, 32046719, 73673971, 31870288, 96927724, 74493483, 489714, 22427715, 79226713, 83914000, 75769601, 32095619, 50971468, 62802785, 196769, 63148192, 14797103, 79505240, 87797056, 77281671, 90209704, 86873514, 74472850, 93552807, 60394770, 31625961, 11551380, 82172479, 88584029, 33202490, 31351425, 58513372, 26448815, 25564189, 88950346, 9567996, 69178825, 97214309, 50372319, 66457224, 63019782, 11966967, 31736341, 75490316, 12817100, 13951725, 56295340, 67281962, 396196, 89132576, 12628791, 15383994]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_c/pspnet_resnet50/rgw_run_0"
