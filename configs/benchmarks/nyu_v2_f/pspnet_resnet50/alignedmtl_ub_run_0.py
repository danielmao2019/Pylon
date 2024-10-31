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
from configs.common.optimizers.alignedmtl_ub import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 68609421
config['train_seeds'] = [52854375, 39798691, 57722635, 37671699, 580357, 2901480, 30201790, 13499582, 27244592, 98813824, 89307889, 18809979, 92412211, 15091358, 12442056, 23452222, 55010338, 55097783, 32329840, 29175451, 45349990, 33612286, 69297372, 6900507, 13438272, 47175777, 11060031, 8710540, 48182561, 41695421, 99689958, 3729713, 33054868, 1228059, 44263998, 43829376, 45741594, 72675259, 43646824, 58275416, 12089007, 63040157, 62417226, 95228464, 22797116, 42407216, 44916513, 35786627, 34636388, 95093795, 86511457, 149579, 84537390, 55869575, 71773822, 55129802, 7549346, 45578758, 92284877, 65914294, 36130944, 88882674, 91190216, 57396158, 90019988, 64749737, 67344468, 18898700, 22050822, 90457285, 5297221, 27072586, 21644961, 22749381, 98969873, 95561002, 27977878, 86164199, 81079842, 40184746, 45501443, 44505085, 39585380, 6577651, 76776117, 18414154, 48967216, 58525332, 55021959, 4796878, 8532429, 82280300, 20225328, 60132320, 22182574, 4057712, 79203448, 49010815, 39617297, 82616785]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/alignedmtl_ub_run_0"
