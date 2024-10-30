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
from configs.common.optimizers.baseline import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 96647402
config['train_seeds'] = [7500957, 16771364, 31448667, 24437685, 18129839, 61983607, 84110024, 56924307, 34235064, 5172549, 82739939, 51731024, 94976032, 35105367, 88699395, 36959305, 55102016, 57539830, 90334192, 35753306, 87659702, 29584702, 55402509, 19139559, 55829598, 57921042, 74963650, 23017098, 55221232, 93018788, 35409412, 93419036, 79134959, 41005748, 43907056, 11022269, 63927293, 45310902, 47148411, 69474593, 96103840, 42364493, 66241120, 81667713, 22441300, 11922456, 92654040, 12805405, 87813261, 27980583, 84711105, 32813982, 66653704, 93975115, 3396966, 52050275, 90661339, 82332314, 14840438, 57602749, 62025818, 39713496, 61818757, 40469175, 43180566, 68984251, 4677413, 77046887, 17638909, 85387759, 47663644, 40037213, 26643968, 24421654, 32053559, 30071733, 38415974, 23942676, 45444482, 3040324, 46375137, 36668092, 40789342, 93348387, 67848369, 84454983, 76763065, 55718556, 15437828, 42779418, 61569409, 57773470, 82754094, 33005059, 52323699, 5250267, 13732903, 39172277, 69926965, 56243644]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/baseline_run_2"