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
from configs.common.optimizers.gradvac import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 25284521
config['train_seeds'] = [26835074, 27617702, 29732402, 57380230, 97183147, 24134113, 14820551, 97195869, 17458525, 28595788, 17148032, 39733189, 48161203, 46561360, 76731210, 83815041, 77247746, 36445195, 36873168, 52953538, 55794812, 55612583, 69045260, 14126250, 2052684, 36095226, 89433621, 82197178, 43295687, 53291724, 6439786, 97985179, 46490363, 84369660, 202856, 53231432, 44128337, 66747578, 35178144, 18052928, 24639592, 44511883, 10988757, 78567530, 42897408, 41347762, 4683931, 38196309, 34451400, 27989422, 77989299, 28655288, 85572075, 43733197, 18401337, 90875677, 87046223, 31660980, 81024982, 8071765, 48472136, 21572609, 98583551, 1372960, 40513264, 93311314, 8725318, 67957757, 10704909, 36945513, 82681475, 55773072, 25490459, 52191712, 78043879, 72201894, 89927954, 61137631, 41643821, 21691831, 58854544, 28333570, 64090191, 26448833, 65276814, 39811699, 68358357, 20452900, 23678724, 76375068, 79024942, 53948626, 36397545, 87497833, 43341639, 15981420, 6636082, 16637308, 6115977, 1663966]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/gradvac_run_0"
