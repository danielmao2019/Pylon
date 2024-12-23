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
from configs.common.optimizers.alignedmtl import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 49180753
config['train_seeds'] = [78948143, 42909566, 12708395, 69059973, 24465617, 2800787, 45323251, 84883229, 25934381, 58981310, 22371104, 14595584, 21891057, 5356945, 63733311, 96072870, 24797139, 34549510, 85436081, 9051831, 45772097, 94744252, 62520393, 85352953, 82027318, 33257967, 87564707, 19809923, 20799989, 21568629, 5761310, 92926762, 80735945, 55561526, 86513623, 69734092, 32247915, 77292779, 58952288, 28583110, 84541237, 57664817, 36128153, 28461106, 54080567, 36630755, 59755039, 9327945, 52988252, 56991823, 82110667, 73439460, 57034505, 3688418, 54322765, 1840158, 84184695, 37461537, 29798950, 52005059, 71626161, 87143552, 26696670, 70871685, 96899696, 38520077, 42628372, 64950367, 55154120, 78082743, 83246806, 2770536, 56339927, 14836678, 12071967, 40281951, 77087024, 13079217, 89434903, 82906769, 23218081, 54342529, 37566696, 56619493, 46908873, 96422401, 62734317, 39027827, 81143618, 81528487, 81600681, 33787314, 76367604, 82965444, 78459900, 31774481, 38185537, 4220212, 56390520, 35387121]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/alignedmtl_run_1"
