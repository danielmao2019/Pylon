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
from configs.common.datasets.multi_task_learning.nyu_v2_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.alignedmtl_ub import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 40166136
config['train_seeds'] = [51574716, 16551297, 8474190, 92845968, 82005460, 53028259, 12607976, 20168648, 29005798, 22176896, 8291183, 8219565, 30998618, 8895837, 42021576, 61754186, 36918522, 90916546, 79797089, 96070036, 58753711, 4268020, 6008110, 81447760, 96006563, 15122083, 10008466, 93922847, 54627531, 1079198, 5722241, 82051104, 7484173, 32048476, 39111192, 86055967, 19963079, 20928548, 80973291, 51220816, 55106830, 61489792, 77143738, 32607049, 98066756, 70087847, 84022632, 12364998, 52096412, 9835152, 14535167, 90639693, 53768502, 9152346, 93885519, 10589665, 81776035, 74594910, 3386340, 34373711, 11546422, 62564218, 42602583, 86485189, 99497123, 37005095, 19907442, 92785682, 63291665, 94983529, 58753954, 81734264, 82584693, 45755111, 13457836, 62430865, 40721189, 66611755, 22734299, 25384639, 97486912, 28775362, 5067519, 8822786, 6178657, 97698252, 56978450, 9355088, 46798326, 92122661, 60510253, 77159131, 62018182, 31494788, 9409970, 94388791, 90020182, 88355875, 74557674, 63419147]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_c/pspnet_resnet50/alignedmtl_ub_run_0"
