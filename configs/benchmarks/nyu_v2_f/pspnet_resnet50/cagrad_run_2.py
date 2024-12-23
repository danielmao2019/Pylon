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
from configs.common.datasets.nyu_v2_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.cagrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 31059725
config['train_seeds'] = [50372723, 50031271, 57045351, 66195937, 98237966, 51701371, 77622123, 63527231, 38369251, 24021892, 74686480, 93690075, 47086759, 52464744, 86786477, 37298565, 53307364, 75341782, 4460655, 82662510, 75869532, 55809943, 84268471, 77215317, 44703688, 38408706, 38379227, 47110443, 64390724, 44190261, 3672832, 16062194, 79283222, 73578521, 99692462, 85180672, 35117420, 34436396, 68367661, 37714274, 36803764, 4766460, 74344927, 63175158, 55333883, 15164992, 88674680, 49927581, 23493200, 96031043, 73936904, 2143705, 94730650, 45205290, 66832152, 72374202, 11557141, 50950607, 31177740, 65253678, 47565549, 72126409, 28304667, 7338469, 6626100, 7531868, 1279543, 47146052, 52177950, 52573392, 84988407, 91603472, 6560860, 49216945, 73529059, 75937870, 71333673, 3443105, 15860635, 61976636, 90875382, 64859281, 66384304, 92833167, 68493122, 22457088, 17871814, 86243532, 78742758, 44685272, 54150787, 55743790, 27209525, 60748869, 4084645, 3836814, 70838238, 58476985, 61536494, 84403885]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/cagrad_run_2"
