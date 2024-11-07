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
from configs.common.optimizers.cagrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 91828623
config['train_seeds'] = [95171700, 83793445, 58510124, 69380892, 91373596, 76734041, 1072908, 21064417, 45368030, 31088323, 75631153, 12089301, 26484254, 29510924, 23354757, 15151891, 2777961, 83256354, 73016129, 52491834, 59938910, 15407495, 6386998, 68373333, 25188969, 30295136, 59724047, 58716647, 75358535, 56819053, 35430417, 33831164, 95130562, 85314661, 41325736, 36775936, 74340991, 6132950, 79187626, 32095123, 86395830, 21940551, 62977229, 63441806, 44339078, 71164649, 67546162, 7045293, 56149627, 30668422, 45737992, 1701746, 23209810, 20380964, 4123522, 83954258, 18635467, 9431441, 16073961, 45677894, 12377913, 85929263, 20898776, 33306275, 10318524, 89129951, 70691063, 77223968, 91959701, 294721, 48806025, 15067287, 89254750, 30436117, 79383088, 89609385, 44315150, 77436881, 96770155, 84399531, 48648453, 79634072, 83769681, 72206366, 10512334, 61033848, 59142775, 86038385, 68749252, 22998304, 83687357, 97537674, 99843475, 84206870, 24346364, 3548400, 3740077, 71595873, 93125508, 95525316]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/cagrad_run_1"
