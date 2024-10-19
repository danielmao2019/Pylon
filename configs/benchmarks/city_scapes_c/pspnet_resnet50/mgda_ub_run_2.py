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
from configs.common.optimizers.mgda_ub import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 94450556
config['train_seeds'] = [32099604, 23024812, 10213102, 30087779, 40157114, 84270718, 73616714, 34647193, 12818558, 24053812, 73498899, 72332180, 18285004, 9492933, 45728892, 10658219, 61755640, 8566962, 60579297, 63811256, 74620481, 89036003, 88525330, 7844272, 95310206, 78250701, 12534854, 97782973, 36391105, 54646489, 75160921, 81298149, 19470905, 76374738, 3553946, 63465321, 61279913, 2167589, 1099165, 51324032, 50852169, 43421295, 57348453, 19279972, 11192903, 66254266, 5515316, 77448741, 99299353, 68403273, 76834660, 57403937, 10628773, 75001883, 40145575, 28759407, 48386170, 59975387, 15835590, 80321627, 83483366, 93013923, 59477573, 84075844, 1282372, 10819280, 93558251, 26438280, 67872771, 41686213, 35462767, 64745209, 55966480, 6001277, 10206286, 29649384, 14779843, 16037533, 60872934, 80233648, 65570541, 55224648, 83468881, 97630941, 57512028, 78920118, 9784417, 86748343, 68172754, 61144837, 91124896, 82560502, 55786296, 30609116, 74787288, 98469008, 32059445, 34502617, 87612856, 8069189]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/mgda_ub_run_2"
