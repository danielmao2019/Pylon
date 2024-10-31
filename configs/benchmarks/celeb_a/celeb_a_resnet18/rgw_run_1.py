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
from configs.common.optimizers.rgw import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 71322459
config['train_seeds'] = [98055522, 41843432, 9199034, 71128330, 24201785, 1954425, 72055999, 46862498, 91058914, 42537830, 62202070, 4648851, 67812370, 6717868, 46051811, 17950184, 32700601, 98700645, 83417810, 46750863, 33421441, 40585547, 8668852, 68521670, 64360025, 22169070, 46001313, 45320606, 68768813, 60099612, 91678304, 62382122, 34694114, 72623495, 93903081, 91220675, 81090220, 96839822, 538856, 75964589, 12363234, 54583498, 84387318, 10230637, 79206607, 66642256, 23160891, 24750932, 54073503, 87780657, 80415816, 32889183, 12095834, 10723353, 47935973, 33554988, 15460843, 14559554, 81356555, 36398355, 15780416, 22069766, 41971573, 20759593, 17452622, 11615921, 9943380, 38338466, 24943224, 20893417, 41587487, 56733224, 88689327, 12716109, 69845739, 78789865, 61673112, 90567954, 49659217, 97199149, 15069587, 39124732, 35768960, 75276899, 35202418, 79531789, 85570820, 32226316, 94760136, 91666394, 41981325, 10079308, 54394869, 24232184, 10996899, 19507825, 22787914, 47638385, 24170241, 61353165]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/rgw_run_1"
