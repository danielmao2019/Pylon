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
from configs.common.datasets.city_scapes_c import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.city_scapes_c.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.alignedmtl_ub import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 31951895
config['train_seeds'] = [59557175, 2564309, 99747717, 69374768, 72676639, 55592856, 42415510, 14620088, 52149780, 21033698, 5775258, 21526191, 78791958, 49260138, 44790742, 11245744, 88798869, 14160545, 73544451, 88650098, 46794407, 10071330, 11822173, 50889497, 94010755, 73896788, 51453802, 31804799, 35225791, 66603870, 35331913, 48756144, 65279067, 45078185, 38740196, 73505721, 79097727, 91129397, 24699675, 65518836, 52375041, 94949567, 72223577, 77938307, 20647793, 90053919, 67608163, 96807266, 50760097, 38417060, 40487501, 4047493, 2380340, 26436877, 57347125, 58958778, 95271346, 67177458, 88570806, 20973414, 7099051, 79340969, 36791679, 69439038, 68533475, 74783597, 20585889, 14267570, 40875236, 23475263, 77614607, 94784415, 24454280, 15117514, 79333014, 13026225, 98346733, 74247378, 67183986, 50586867, 92768148, 97527740, 46011560, 32191444, 48125632, 59159210, 67128808, 31027428, 57156383, 74656092, 13683968, 68141601, 8111623, 37702889, 82212254, 66230914, 7534688, 89479892, 3998811, 22970128]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/alignedmtl_ub_run_2"
