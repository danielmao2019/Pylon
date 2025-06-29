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
    'val_seeds': None,
    'test_seed': None,
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
                'class': schedulers.lr_lambdas.WarmupLambda,
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
from configs.common.optimizers.multi_task_learning.alignedmtl_ub import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 21034248
config['train_seeds'] = [15152491, 64651028, 49117974, 93649408, 86172190, 2581435, 93416157, 85954744, 53001629, 80334162, 61672906, 46502190, 55962333, 26789760, 18239807, 87981127, 74652744, 16737341, 77613692, 78552746, 4949483, 36395333, 76962308, 53175623, 31233277, 84152244, 40649576, 90542376, 74158870, 43123448, 45108687, 93394493, 196486, 49790431, 67054384, 90734776, 93325194, 1836782, 64675636, 83390716, 78471508, 35972204, 78554599, 4982809, 83616215, 576949, 43489606, 98829785, 21444259, 26964337, 25916434, 43634481, 89692211, 15300977, 6081819, 18534985, 15372911, 3154093, 33966081, 54336290, 36442255, 78825931, 12763021, 67759622, 3254329, 86386360, 6727830, 97414295, 80480574, 85245598, 60856559, 44760803, 58394230, 57781044, 91375742, 83047803, 86153375, 7281705, 1556062, 59404034, 70699800, 66457086, 64939613, 17229218, 9141378, 69274797, 28356572, 6744153, 62938495, 9508926, 34293099, 13193311, 7786817, 68021936, 34662776, 77275120, 5635398, 76076673, 22945426, 14917947]
config['val_seeds'] = [93216308, 65122366, 42650422, 62463516, 63094494, 71206473, 74849136, 47536808, 55567186, 83503671, 46760366, 68752358, 94218074, 763665, 76672197, 62634791, 30269059, 39742916, 11336665, 75876896, 40961281, 8477765, 92740415, 11092204, 80505212, 14133548, 28318479, 72876971, 70809141, 81270466, 18598139, 65842337, 14142415, 28612456, 24425276, 21508153, 71374013, 28730883, 56963698, 65770823, 71036849, 54801968, 79184198, 93017881, 49593518, 73477937, 69498859, 79197509, 94895255, 45151877, 34852087, 18533878, 3248392, 70518881, 24874841, 21104252, 67109607, 80657649, 24821289, 44190667, 57186837, 94831380, 83029809, 58119094, 39893033, 24219439, 1367307, 99775833, 14442142, 17930301, 14039990, 19298912, 40579521, 40366691, 85282268, 70816975, 11301217, 9731345, 87610016, 38037606, 79455975, 4672659, 10906465, 64309630, 92462772, 47409747, 18406551, 87817263, 44452765, 91937038, 98079091, 57717611, 99065278, 64217780, 88647580, 27203264, 20352140, 3828242, 73105885, 9870811]
config['test_seed'] = 71462633

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/celeb_a/celeb_a_resnet18/alignedmtl_ub_run_1"
