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
from configs.common.optimizers.alignedmtl_ub import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 37968421
config['train_seeds'] = [20941904, 88355618, 41557998, 78087950, 6019890, 7547882, 66006524, 73028705, 96840923, 57713136, 49325083, 27368732, 24050643, 28449246, 87694736, 38574932, 59336888, 78156638, 21958196, 85293835, 56784483, 94745536, 94560220, 49260650, 86725917, 44167470, 37033723, 59828124, 26750043, 42910535, 12429838, 97598540, 48276011, 50895349, 75364639, 44982890, 11105643, 82862371, 43063158, 37270577, 74004373, 844485, 6512867, 63399597, 3513619, 59173526, 41729230, 69444905, 88905831, 22714790, 43643876, 17656391, 89541950, 14470998, 97024784, 83324699, 22304598, 4117276, 19027233, 46906758, 42462417, 34169363, 77794595, 5410922, 14442958, 41635389, 98713865, 89307705, 95585243, 54542761, 78239154, 6706289, 31264196, 95004218, 50723016, 39461455, 98405569, 85547075, 76230662, 64791740, 17251386, 11972266, 12289803, 9436669, 40306167, 97588013, 58557923, 70446579, 623616, 29992812, 66053620, 80490177, 98197115, 94790025, 17890238, 69895279, 23905132, 10166312, 14647080, 78007898]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/alignedmtl_ub_run_1"
