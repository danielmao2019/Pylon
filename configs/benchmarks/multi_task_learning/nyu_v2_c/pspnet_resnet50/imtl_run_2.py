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
from configs.common.optimizers.multi_task_learning.imtl import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 54109687
config['train_seeds'] = [19802275, 53531345, 44269060, 31121861, 8980634, 30766012, 25833816, 97620684, 76913780, 78213699, 20537476, 7977097, 62913656, 36508683, 49357151, 62450415, 40085596, 20539297, 54979967, 70181865, 38571098, 77453518, 89315753, 16682786, 77237856, 83129067, 44830240, 68203466, 26580963, 39744181, 41593763, 89100765, 86229845, 37640302, 68323338, 18315136, 56734588, 75003508, 29914966, 18462366, 97843831, 89291402, 16404627, 87646351, 3615542, 49131281, 58841802, 88627006, 48510682, 23145081, 54384201, 77060613, 5127769, 86376918, 31276154, 54598382, 59866812, 15329685, 44758282, 71135213, 92992497, 17553144, 116130, 67579197, 12777202, 49451251, 60820683, 93417124, 9258045, 47234748, 95782494, 66489694, 3485351, 41163316, 8623023, 97852925, 42861703, 42712050, 7562765, 73386290, 8812163, 1350777, 25781851, 51558856, 44983028, 32073788, 8870008, 62654994, 83636337, 48008182, 36230330, 85563503, 6996473, 97260235, 32124107, 32136305, 33449911, 65966347, 65666366, 19471789]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_c/pspnet_resnet50/imtl_run_2"
