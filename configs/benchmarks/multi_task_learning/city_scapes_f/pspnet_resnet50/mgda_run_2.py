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
from configs.common.optimizers.multi_task_learning.mgda import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 30678065
config['train_seeds'] = [93883355, 65686861, 18581651, 84676814, 19175504, 67535710, 72578494, 91429486, 40512231, 66567494, 6169528, 22403037, 67661499, 91345481, 38691813, 10503174, 6999116, 97816439, 76070315, 1883660, 15054891, 39442735, 80132474, 12611637, 9536089, 21717226, 87488100, 18617359, 11625507, 37928424, 37933410, 32834047, 32666522, 16270230, 26123289, 68084089, 8716774, 35127199, 30481658, 45697133, 91118239, 34469281, 97441653, 81701714, 9654178, 78020730, 71447807, 18849058, 81742780, 31278392, 76148927, 64996302, 80988692, 39725242, 13613243, 19434858, 42256186, 65228172, 39190731, 83821907, 32392079, 51803508, 70161641, 96488491, 43958213, 19306103, 1988321, 84011203, 19982645, 63939345, 9543908, 96778124, 59459887, 3598504, 24660892, 70337275, 28515851, 93468307, 61927694, 7660131, 30081835, 26177998, 53395272, 79879628, 90838523, 10649390, 24055028, 74975602, 8217367, 78611307, 7218038, 21038787, 4167401, 68852157, 36636714, 77490697, 18132528, 76929987, 34073058, 7768579]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/mgda_run_2"
