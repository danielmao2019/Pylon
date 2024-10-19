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
from configs.common.optimizers.alignedmtl import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 57531446
config['train_seeds'] = [85092283, 85882375, 57565241, 5828945, 91963750, 75926243, 48427751, 71410552, 27008389, 94305495, 65570482, 54758113, 71923125, 67855666, 41584000, 40904005, 93417542, 63060584, 38185167, 99056322, 69154570, 96553307, 37634528, 96538210, 56851730, 47822628, 79805498, 94223914, 62469971, 42495705, 43333094, 47802536, 51107069, 43796264, 27009186, 52964461, 75746977, 58890741, 45805276, 10461367, 5051009, 46306013, 7878793, 36983069, 78208815, 95038779, 95840129, 24331406, 58830337, 83600283, 11382920, 50486621, 34133266, 18087204, 63677174, 66501910, 13605240, 42241222, 2209974, 81161345, 69606972, 43646179, 54573382, 31035979, 34699978, 23075171, 86645855, 47012343, 24686493, 39031101, 47580080, 26318169, 55373876, 85541234, 88333903, 18931707, 6056337, 6558184, 91838442, 32295730, 50020136, 26132975, 13340555, 64425710, 24671295, 74831665, 89955707, 86503845, 80651262, 60794010, 47238354, 76233116, 91767555, 42175143, 56574958, 50798593, 91039488, 11186386, 42358351, 32549778]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/alignedmtl_run_2"
