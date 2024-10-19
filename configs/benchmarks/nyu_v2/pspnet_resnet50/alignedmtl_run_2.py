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
from configs.common.optimizers.alignedmtl import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 73839776
config['train_seeds'] = [75029856, 71720582, 6843083, 13283175, 51177936, 51730843, 58984225, 98896774, 9233318, 63398234, 54530709, 84788867, 12120543, 69967460, 2085847, 65013630, 55342503, 66977968, 53821460, 83407435, 32353067, 27742757, 22329051, 88332417, 46729581, 85858537, 48875077, 43652829, 52926192, 8961602, 93313961, 73017390, 15049761, 42373708, 21592657, 25024224, 28266490, 81247303, 32203081, 14176954, 91758680, 96498607, 18419136, 23443404, 38690373, 76544353, 75644722, 24282031, 63742202, 55125203, 73872673, 55330839, 95678859, 47404793, 19841702, 26229238, 58876020, 89384972, 3986263, 4335461, 90572988, 37377984, 7599023, 7130334, 94030942, 10571762, 96190849, 11007337, 72590788, 84961709, 1157538, 25453432, 94651881, 52668893, 41118715, 95092148, 88704377, 90724054, 71540462, 17639128, 76811261, 13852837, 14067124, 87993677, 39299417, 12805842, 72407177, 39421544, 3979788, 81996694, 13047337, 75987662, 88574672, 45036316, 62328565, 7906187, 73057929, 50869386, 97270475, 72642780]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2/pspnet_resnet50/alignedmtl_run_2"
