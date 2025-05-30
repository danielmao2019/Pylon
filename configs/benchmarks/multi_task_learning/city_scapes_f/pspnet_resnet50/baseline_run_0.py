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
from configs.common.datasets.multi_task_learning.city_scapes_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.baseline import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 70239363
config['train_seeds'] = [36997926, 90200892, 5668546, 2072455, 79231915, 10004440, 40916590, 83031166, 21546881, 68474822, 47778564, 84092490, 81064541, 31599342, 11395251, 34078142, 57487912, 84457169, 26983517, 30051172, 6175395, 75949742, 73278055, 49430915, 42056029, 73109137, 19066347, 21282885, 12820637, 56871951, 68042025, 71585003, 88189689, 39398471, 9933217, 98165441, 70869002, 35661584, 78462998, 16452687, 31701654, 12471671, 60901172, 58945104, 85469720, 52974069, 57184316, 43520440, 24905547, 48259835, 46570142, 42658734, 93952690, 56841398, 44185373, 54884612, 95518979, 28671486, 57012857, 95285700, 20796248, 79720901, 69950922, 51917470, 12121859, 96777981, 62732489, 87614976, 30745747, 93273273, 81143118, 17707715, 40414889, 28033224, 19538435, 86125893, 42394496, 9520389, 66471164, 32260915, 56704789, 58380436, 41273813, 61485059, 3965091, 77510818, 19042832, 28144994, 4329937, 40086730, 36170261, 46437996, 22078280, 44676997, 24637456, 29225491, 49143563, 76753759, 86509516, 7245799]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/baseline_run_0"
