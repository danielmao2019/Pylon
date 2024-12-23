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
from configs.common.datasets.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.baseline import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 30248909
config['train_seeds'] = [17737549, 77789533, 75442135, 94490238, 52339035, 99452444, 42138126, 14079263, 91161309, 52972616, 12682410, 71120522, 20112253, 49451820, 24176449, 47080193, 96904192, 10957913, 67582184, 2708968, 10149433, 15516620, 82528898, 52176564, 67837494, 55198031, 15318520, 17634831, 9499306, 31819819, 88201040, 87953212, 68625667, 21767092, 64280815, 10684760, 35743323, 25486338, 71726791, 29936853, 4938760, 40518776, 15190168, 59837780, 54560667, 93060487, 74780225, 49846607, 31032348, 1720105, 36537323, 50959296, 77346784, 96734468, 32565006, 32211346, 94514590, 70661197, 2502278, 44458889, 93957993, 77713019, 83834226, 11697888, 64557113, 59468796, 52518845, 49922668, 1041114, 24459395, 18703954, 23453151, 5867820, 94446779, 37572190, 94270183, 47956528, 44231333, 14181980, 87721420, 79673985, 89048291, 1458297, 64750016, 73536901, 7173936, 64766857, 45924686, 37922395, 19555468, 73065974, 52632816, 88991310, 87236624, 64349654, 91037140, 45138833, 84061370, 25295907, 29228262]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_f/pspnet_resnet50/baseline_run_1"
