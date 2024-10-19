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
from configs.common.optimizers.nashmtl import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 65863947
config['train_seeds'] = [16118002, 85513622, 67021092, 70122426, 1634036, 6715080, 70294810, 44641096, 31392236, 87968785, 76724098, 62041641, 17923793, 96192773, 4713462, 96206795, 64405273, 3903469, 94406154, 81318523, 56288640, 1816874, 98505775, 88290978, 20362208, 46270879, 24550250, 96264596, 34081772, 76236087, 65158772, 44097787, 87682273, 58478322, 9497795, 35585308, 52847196, 69447866, 76646281, 27037694, 21467748, 92490846, 97019166, 39108518, 74617789, 48584423, 7146116, 35436791, 49410307, 45062205, 79850229, 82922506, 85657108, 24190175, 64933383, 79082752, 68201987, 48121637, 76965355, 58437261, 10750385, 35773951, 67284173, 66065103, 28446245, 56931077, 13730031, 43774103, 34144854, 81328400, 97623806, 85824733, 62726903, 6738438, 86737047, 460477, 49690621, 2690011, 24518546, 44560681, 57525290, 28180884, 12065154, 47423582, 76715150, 93764300, 35139670, 75905749, 63852830, 32552375, 32142459, 89082039, 44885966, 41696293, 3403432, 36436481, 80584462, 14557736, 79927101, 76293632]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2/pspnet_resnet50/nashmtl_run_1"
