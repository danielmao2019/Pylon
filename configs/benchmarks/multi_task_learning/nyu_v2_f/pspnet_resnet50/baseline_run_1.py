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
from configs.common.datasets.nyu_v2_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.baseline import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 72936230
config['train_seeds'] = [27917203, 34220350, 69373811, 85194049, 39397757, 37448772, 63382763, 94493889, 83855604, 4803496, 73519377, 70007803, 34765140, 56899933, 37268955, 99812998, 5074523, 86912782, 67146503, 53097911, 97379723, 1328666, 91381347, 91952303, 17102531, 37689312, 99465431, 12129734, 92390515, 2665641, 19995932, 14440912, 59645289, 63754603, 63602564, 19876281, 99759722, 7970466, 98238085, 26185619, 67867219, 77125701, 65383784, 60364605, 28447668, 5216974, 48344662, 34702910, 18175524, 36171801, 65845182, 88749282, 79773706, 88509662, 9516161, 26608923, 86519423, 38205631, 29253448, 91476057, 78659855, 21442310, 82007665, 46384150, 23332792, 59537210, 43922126, 79692277, 12143707, 53553112, 39908519, 93807243, 37325637, 76517945, 29930047, 82283834, 98438313, 59771756, 14678593, 88576782, 45755749, 5785196, 91231484, 28100566, 10621231, 78479865, 56625521, 38568632, 19397909, 78692305, 724289, 33824173, 2620634, 93331479, 89505482, 68543923, 54605650, 82414432, 88148973, 30587317]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/baseline_run_1"
