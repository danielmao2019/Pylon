# This file is automatically generated by `./configs/benchmarks/multi_task_learning/gen_multi_task_learning.py`.
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
from configs.common.datasets.multi_task_learning.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.cosreg import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 75322215
config['train_seeds'] = [53102936, 64688092, 47899272, 97426545, 28146629, 62227386, 54247730, 19212528, 49384037, 75838374, 97316431, 48966426, 11158242, 45316587, 76858756, 9851074, 10762507, 52314656, 83776491, 70635296, 93376994, 39451410, 9356088, 4433467, 10601825, 64733267, 2757724, 42145858, 31658922, 51844936, 19575974, 80473072, 54302845, 63246515, 14332792, 47303400, 15300870, 78928946, 37134342, 37314814, 36791960, 74174476, 99612831, 92831569, 64187990, 63259705, 84727531, 48856239, 76723796, 99316059, 46302316, 21904628, 18978054, 20907039, 88663899, 19125111, 56222434, 97524759, 92017911, 67666703, 71566088, 46303064, 61871281, 28275900, 52048870, 79845104, 82295613, 84306274, 69912969, 17424446, 46307844, 98860888, 87349041, 7028700, 23454875, 24022112, 95961817, 6123947, 43712581, 12260171, 36422167, 19874, 14093223, 58395519, 67454590, 94024117, 83419361, 62367796, 70534397, 60991759, 32547845, 61842270, 63246486, 49229991, 2852105, 21374933, 87437217, 40394664, 21835638, 14244210]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/celeb_a/celeb_a_resnet18/cosreg_run_1"
