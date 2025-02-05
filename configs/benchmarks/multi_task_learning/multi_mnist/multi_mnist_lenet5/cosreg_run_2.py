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
from configs.common.optimizers.multi_task_learning.cosreg import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 94448347
config['train_seeds'] = [58319614, 15402660, 69338668, 61859880, 94195557, 66163777, 48340113, 72993200, 3537830, 21072724, 8573851, 69274992, 78929584, 57751436, 7647996, 81211296, 2538015, 59677088, 66716360, 37689756, 62261105, 14627233, 88376221, 30785890, 37701875, 86608170, 78536537, 13721060, 31516897, 58588780, 15601115, 38411416, 58267436, 31983999, 60065879, 87905799, 12146438, 1566168, 47864580, 4523973, 85541874, 24063272, 34587281, 6344806, 94783824, 81728407, 64993805, 3746122, 59404902, 72120317, 57705778, 31261750, 65197735, 6244618, 95795256, 82453666, 63529102, 69889259, 88174880, 40873649, 15966091, 89050602, 77821657, 44544453, 54312852, 5129436, 3510533, 75345112, 84682482, 45546975, 95200054, 61016225, 10144847, 97618360, 87378914, 99002194, 82346763, 88743494, 37315216, 5354139, 70002668, 70489648, 80688230, 77588335, 2174078, 35254036, 59146791, 53580712, 56856390, 79222489, 69604829, 57560053, 69674100, 76380240, 25798845, 32259199, 10132955, 50237340, 76666802, 38038217]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/multi_mnist/multi_mnist_lenet5/cosreg_run_2"
