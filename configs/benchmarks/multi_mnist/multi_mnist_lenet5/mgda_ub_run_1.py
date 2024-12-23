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
from configs.common.optimizers.mgda_ub import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 98699233
config['train_seeds'] = [97817798, 77854522, 20701937, 72260033, 50668861, 52419412, 70454660, 71485197, 12409422, 51846893, 29912933, 90270124, 95400430, 3583541, 51973284, 96832459, 31533980, 26920748, 64847407, 56015073, 39775472, 64771829, 25529979, 80726351, 42678956, 590970, 12420993, 45266069, 87583032, 86875958, 97027496, 18570236, 41878245, 39191207, 17419722, 70507654, 85015721, 6866226, 63539047, 55666000, 51212628, 52971753, 66541457, 83028878, 49571694, 80529327, 95361117, 53361300, 43249601, 63363804, 25073275, 1132931, 55114860, 75368046, 19076588, 78319114, 68458238, 14191257, 58312431, 71650702, 94544690, 22217110, 58199057, 55489951, 81090101, 62787576, 9974095, 79174825, 36873925, 89248069, 15934662, 8661808, 23716775, 456325, 55803249, 66141346, 28294949, 13191732, 37421854, 49278153, 71778800, 73361736, 67122801, 69524882, 23297931, 66850163, 4087171, 60023619, 91462023, 45864, 52453292, 42629545, 27478529, 78934670, 91617358, 90470003, 46058222, 61634167, 9341355, 12690896]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/mgda_ub_run_1"
