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
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 47521027
config['train_seeds'] = [49045350, 32842266, 4438290, 94173391, 419759, 14627119, 37128038, 13925830, 27866679, 57290548, 649444, 66485433, 45652661, 92788470, 45848997, 50758412, 24170042, 80020976, 54081462, 61585615, 4556564, 98237046, 63868769, 91573438, 37621133, 43441189, 33700744, 987074, 23233535, 29407150, 86661281, 36483080, 73312482, 88358226, 50220993, 82767887, 32702263, 36796267, 30678287, 59253311, 57339526, 19276912, 97677131, 36292396, 71539970, 70032824, 42184828, 54561465, 36183847, 71555991, 49049975, 80050994, 88336070, 19332967, 16415266, 98317058, 54822542, 82775206, 12996866, 70162400, 26610868, 74051918, 10851530, 66486369, 66963175, 5017855, 4609890, 77943109, 72042457, 67007770, 37394677, 35108884, 87480495, 9833883, 40336066, 80893586, 69634754, 65514449, 52681237, 46030645, 79261813, 46179024, 17393962, 55033592, 5101438, 2033785, 23146008, 72732584, 6911500, 46283537, 97317103, 53544400, 69669336, 20315800, 36235983, 43982193, 91670547, 83355417, 23565273, 50377952]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/cosreg_run_1"