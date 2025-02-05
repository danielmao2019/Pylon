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
from configs.common.datasets.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.pcgrad import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 40994442
config['train_seeds'] = [15895446, 74081808, 51856937, 24130338, 5278865, 6181091, 97409603, 62752755, 72982821, 86950074, 12883048, 72411267, 80424926, 91130133, 49634304, 67542257, 89016744, 52543938, 47597529, 95483875, 53059563, 42457525, 66527550, 21946586, 46741617, 84307496, 55926356, 14195924, 78115961, 66543152, 25813087, 5057152, 57039552, 66129255, 81544373, 92988992, 86320934, 69782150, 45146782, 41804933, 55115001, 53952450, 99046874, 59834004, 5801146, 43722716, 97639953, 50435960, 96937585, 3862080, 37781581, 93492784, 16587090, 93580434, 70632060, 32934735, 53416500, 62565126, 74213430, 68925395, 37391597, 13095155, 17986430, 800992, 62756972, 30651199, 10090600, 63228808, 8813930, 62630426, 84058782, 9053241, 8683671, 40188190, 56751790, 48966600, 57340281, 29251488, 33761601, 25327443, 69131599, 9894887, 58951083, 71605876, 18278335, 19022849, 13076723, 59535864, 81754792, 54699847, 75414907, 13534629, 49195205, 72812613, 32807619, 37415591, 90533997, 37851000, 82569513, 71265419]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/pcgrad_run_0"
