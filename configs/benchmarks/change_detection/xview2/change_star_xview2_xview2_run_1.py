# This file is automatically generated by `./configs/benchmarks/change_detection/gen_change_star_v1.py`.
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
                'class': schedulers.lr_lambdas.WarmupLambda,
                'args': {},
            },
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.change_detection.train.change_star_v1_xview2 import config as train_dataset_config
config.update(train_dataset_config)
from configs.common.datasets.change_detection.val.change_star_v1_xview2 import config as val_dataset_config
config.update(val_dataset_config)

# model config
from configs.common.models.change_detection.change_star import model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 77242167
config['train_seeds'] = [31669001, 39273642, 23863547, 98017337, 82135406, 79321325, 342951, 27164558, 46068032, 4476766, 57915910, 27435580, 22975195, 14854470, 12711220, 32650604, 45795813, 26262317, 48294652, 78626781, 65751242, 39761090, 32804302, 88652354, 92848858, 81506743, 95151014, 63444808, 88963021, 19994217, 56152070, 83508653, 54834315, 2901354, 20564974, 81731905, 47561183, 10847040, 67447127, 34408348, 30737505, 89614124, 43964001, 64448071, 64699930, 56886177, 27678716, 37918362, 4456320, 94446999, 86907023, 94070348, 32399621, 85634861, 6971706, 98664349, 31731036, 65609610, 42196019, 51716582, 70090528, 58380759, 45917803, 11112547, 23969112, 40228954, 35907418, 82822598, 12137601, 21435017, 15851233, 70031618, 64555659, 61415890, 94253301, 62475847, 54176565, 68058967, 7521210, 14118191, 46449691, 21464747, 45860953, 23474025, 45254424, 98093547, 60939360, 21242929, 90069848, 69339895, 69365308, 31114114, 81457254, 61653922, 44735264, 28404607, 23845790, 12409353, 34522726, 85783613]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/xview2/change_star_xview2_xview2_run_1"
