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
from configs.common.datasets.nyu_v2 import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.mgda_ub import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 21859998
config['train_seeds'] = [45344844, 11803782, 3900409, 78634861, 67821404, 26390152, 9181939, 71880338, 81221692, 17670096, 93075653, 89673394, 14342031, 63098361, 97772143, 20911097, 19642132, 23538249, 74330903, 11854100, 89848006, 21764918, 61373018, 78084545, 56931945, 72128484, 63673241, 54407991, 94790329, 53828775, 70155931, 31472414, 37832601, 20165120, 40769272, 86189565, 37527233, 76711667, 56610071, 84821794, 35917473, 19704377, 18481926, 21767993, 55629830, 71391645, 92131436, 57451756, 43101054, 99893415, 89459505, 65287268, 35939661, 42962603, 77079634, 46537116, 6015063, 40048309, 16273182, 21428238, 3695646, 5745383, 87753800, 29356168, 18434779, 35667574, 59758178, 29000273, 34954665, 65653346, 61189040, 22511410, 45423066, 66184773, 56444924, 23718870, 44302275, 36017711, 32658138, 91013599, 26174239, 66490696, 19843789, 61608243, 18633431, 13563415, 75880357, 74201577, 21554089, 84572350, 1384447, 23787822, 84410170, 24119728, 42107159, 8634265, 84627288, 64603787, 17035995, 56022039]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2/pspnet_resnet50/mgda_ub_run_1"
