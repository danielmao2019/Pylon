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
from configs.common.datasets.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.cosreg import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 30701848
config['train_seeds'] = [64690639, 31309133, 27649167, 31907680, 95150992, 38845276, 55118615, 56428916, 78382066, 97249975, 10901351, 62949492, 38295631, 32600240, 96270940, 20871549, 68192939, 52856561, 84350610, 92400110, 17132228, 29779704, 84734261, 11912132, 17342956, 94443065, 3918135, 9697194, 96946932, 41449095, 63054393, 89706160, 78826466, 47134626, 76579962, 96817168, 85112893, 73670831, 47519935, 24007067, 4270282, 94517575, 17268859, 13621928, 90378206, 76295829, 33920307, 92024367, 40702737, 17486641, 48239787, 43491319, 83520387, 30808734, 42046558, 39316860, 239377, 76803150, 66485498, 59726472, 14915931, 65840848, 43374879, 46503220, 73423380, 11704001, 33248598, 28515176, 9219185, 2714951, 50839500, 14119329, 99351124, 33503018, 5030116, 18924588, 811887, 18349742, 23501868, 65338532, 72250500, 42123646, 95254603, 86703171, 98738932, 92703075, 59135752, 74012513, 82511762, 61798702, 96603270, 20608946, 8823629, 12309682, 95390957, 95152046, 66815745, 83667579, 78888024, 73016340]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/cosreg_run_0"
