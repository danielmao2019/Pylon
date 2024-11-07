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
from configs.common.optimizers.mgda import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 58321932
config['train_seeds'] = [99875254, 30680030, 25321009, 62794269, 75701282, 20495359, 46642198, 51930345, 59982015, 50695481, 21275790, 86585334, 81087762, 52165275, 69279042, 55907708, 59600414, 30766462, 93692936, 73589055, 97306818, 76781797, 40564294, 26715324, 21071862, 27225666, 60435474, 8105727, 22338265, 22069360, 44203647, 46813206, 41866749, 30735443, 93493013, 59332993, 45264300, 71843516, 11692057, 96906083, 96529972, 57148233, 69177520, 6311767, 71823476, 73043089, 21525599, 78209545, 59505793, 56018094, 18989336, 89773317, 99309339, 28115603, 55991799, 14253624, 84421268, 36453913, 17497093, 76243278, 70835924, 27714460, 42508838, 65510429, 94423508, 81086931, 78049872, 78679819, 54489887, 46292485, 19148810, 70583731, 78079430, 65964453, 28174233, 31409148, 17423967, 61116026, 36045257, 59568955, 20470438, 55036988, 48656092, 72858056, 23069844, 66037061, 40433374, 72436680, 80024363, 94715816, 47187659, 33514032, 27701040, 24084983, 99367424, 57334367, 69154722, 44337549, 2670892, 40841271]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/mgda_run_0"