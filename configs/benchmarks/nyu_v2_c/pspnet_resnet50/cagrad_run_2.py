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
from configs.common.datasets.nyu_v2_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.nyu_v2_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.cagrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 42720820
config['train_seeds'] = [93750716, 95419296, 77461787, 89034291, 71088810, 48573337, 14863826, 2702669, 13935647, 29033497, 93652549, 52039038, 4712741, 43966557, 21286607, 63761589, 1662875, 18046194, 54808893, 43974427, 31773344, 86641978, 50155170, 36051051, 85309799, 85285097, 72149537, 57095326, 45520995, 13196145, 74105737, 61645073, 37616704, 35071734, 40762528, 80690107, 89516698, 41640110, 71803368, 99887800, 66295623, 50618469, 74098326, 49897503, 11087144, 49351932, 79480452, 86470873, 18764997, 46624689, 26635443, 55645772, 55195090, 19068289, 82898943, 69531016, 1342031, 29869621, 58054494, 91318436, 61858275, 2884057, 42837112, 42602525, 56688717, 72263908, 15146255, 31602467, 37304488, 80113284, 36063873, 1828167, 65657238, 91333539, 13800278, 78173936, 81477392, 71535519, 82898197, 4040483, 47116676, 32110023, 81134772, 32744719, 11677688, 1085129, 5298014, 74176840, 23483421, 22302398, 96401266, 65421734, 92713120, 81311745, 37963345, 92509511, 86490748, 24423883, 91355094, 41141501]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_c/pspnet_resnet50/cagrad_run_2"