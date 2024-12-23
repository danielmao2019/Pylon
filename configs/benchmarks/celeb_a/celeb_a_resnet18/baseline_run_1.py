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
from configs.common.optimizers.baseline import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 46276709
config['train_seeds'] = [28372915, 8780058, 93538326, 92639270, 16285440, 39182636, 97971598, 19727089, 44481622, 34669639, 59314299, 19922526, 72360280, 55807732, 66859042, 80361917, 9482333, 2764221, 43677129, 49849374, 61193702, 86229126, 44745024, 64521139, 73572833, 25966531, 51490254, 3738384, 44380081, 71957999, 17790968, 58457916, 35947055, 85354263, 42082607, 63248684, 48751223, 92630133, 62516998, 2260347, 8878269, 30130079, 92969412, 22564606, 32926777, 47557920, 10748618, 85655501, 18260853, 80687204, 57019472, 71895609, 70308403, 13475358, 74297173, 61958247, 58676251, 62154731, 40179532, 66987374, 8822455, 18579825, 46269461, 17127388, 93594049, 61674662, 55159848, 78175794, 87290760, 18965254, 16894746, 10443906, 39273756, 98166103, 36197148, 93283963, 15370066, 91155553, 39354116, 82867686, 47395921, 12458251, 8030466, 5969196, 49514489, 72913752, 55405258, 22633867, 14118029, 40092403, 8307632, 19517495, 58352498, 92170097, 28551723, 81946829, 87096343, 26745597, 70029851, 28265702]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/baseline_run_1"
