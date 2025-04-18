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
from configs.common.optimizers.multi_task_learning.mgda import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 33915386
config['train_seeds'] = [39756991, 8719138, 7915283, 85659220, 2534616, 8983147, 41932440, 1120957, 1733898, 37273556, 95524538, 86662155, 89156261, 71772583, 20102686, 49995531, 94883630, 1085909, 30367956, 80995061, 25672299, 82501560, 4509669, 46043138, 1854613, 25093522, 16289093, 17335848, 37532466, 62671803, 65053912, 93019614, 60568299, 62259087, 50178830, 9125157, 17922865, 76572407, 96824850, 39701222, 91305376, 67656609, 39765534, 57619351, 80365324, 49700091, 29125223, 22165396, 42611398, 97982130, 15494446, 94132851, 68476358, 37148388, 1700205, 90656312, 23733929, 43236224, 30753004, 73199069, 94534098, 52127998, 39581951, 82333728, 33368581, 36568817, 10976222, 4253456, 72699884, 86069455, 96263258, 12712160, 30218712, 82623665, 92968853, 47536356, 98760933, 6475533, 33918503, 20139112, 97047734, 75975061, 86819289, 42677445, 23896432, 98681402, 49137239, 15817777, 9808268, 512339, 6189591, 8196423, 16844291, 49136045, 4073794, 61422800, 80454572, 73452890, 43220899, 16635]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_c/pspnet_resnet50/mgda_run_2"
