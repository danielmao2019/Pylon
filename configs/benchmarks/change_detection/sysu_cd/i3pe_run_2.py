# This file is automatically generated by `./configs/benchmarks/change_detection/gen_sysu_cd.py`.
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

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
from configs.common.datasets.change_detection.i3pe_sysu_cd import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.change_detection.i3pe import model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 30070509
config['train_seeds'] = [66706792, 57134929, 92474690, 86537101, 28128978, 57360167, 31009394, 11786558, 14432248, 65071380, 33724409, 7432459, 84497618, 56014884, 5903580, 80512637, 59516738, 96722648, 65939932, 144389, 11521204, 38797862, 26821586, 76051666, 94522788, 55973557, 58676557, 5306850, 70331610, 54956644, 15198187, 67235369, 4253655, 67946489, 74172997, 98136123, 99832670, 14959083, 80119174, 94894639, 65207472, 66539342, 80347333, 22576322, 53077860, 90877862, 30580910, 96529329, 16174934, 66718753, 90359655, 22345614, 95606420, 69647346, 79785909, 70074933, 6895233, 1303622, 93652135, 80911599, 46673382, 1153103, 99638266, 43707087, 67808583, 94918479, 1948568, 45392463, 90636426, 79977766, 78593094, 52219640, 75376089, 25002841, 17371703, 26896469, 30265848, 48562179, 87463141, 80583321, 27354209, 43714001, 2347152, 5584633, 95766893, 15521660, 80013719, 628961, 30981811, 53203751, 59396326, 95444367, 24009329, 92405868, 69606599, 97115767, 90016510, 15000065, 47482812, 56685011]

# work dir
config['work_dir'] = "./logs/benchmarks/change_detection/sysu_cd/i3pe_run_2"
