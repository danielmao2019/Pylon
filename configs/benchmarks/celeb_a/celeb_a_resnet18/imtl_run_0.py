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
from configs.common.optimizers.imtl import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 10166749
config['train_seeds'] = [23290497, 89192485, 10856944, 93080796, 64002810, 2221452, 28692698, 77185271, 1480161, 8683353, 43604149, 84461467, 48078638, 96452679, 40322467, 96443973, 99808740, 3774050, 67526311, 5458661, 837277, 13424291, 41433231, 9341870, 30157172, 56058445, 42630293, 35749839, 28023534, 56111901, 66298994, 37314723, 53606480, 40480732, 8561971, 62626616, 84788161, 2488577, 29551184, 41358832, 92833101, 50856591, 31390261, 23157160, 31327731, 99349893, 62540565, 122381, 81091903, 66816705, 86274805, 17747044, 99929469, 75636867, 59709922, 72374318, 12887617, 25664107, 10073439, 37597732, 90943525, 27262137, 39105411, 99285978, 90198991, 18618204, 32453021, 92619922, 52300508, 30542196, 14625528, 68784884, 87139508, 97365155, 60470188, 70245806, 68218586, 96337321, 23984365, 47047734, 18665680, 25880781, 19773724, 40354696, 47086890, 70692985, 99511726, 70597296, 91444008, 6735063, 25004484, 58865986, 47853807, 70991446, 3019298, 67164455, 64393297, 74789178, 75866032, 91039362]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/imtl_run_0"
