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
from configs.common.optimizers.mgda_ub import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 8453159
config['train_seeds'] = [45390237, 87270645, 71034934, 76290651, 25249886, 44269079, 81821861, 6312803, 42834790, 55877815, 78638490, 83569610, 86368292, 87451429, 74755663, 83705569, 67530934, 99983568, 75092630, 24948466, 84835215, 26685341, 71817632, 34437845, 11175113, 19902287, 49222676, 38737167, 77107759, 59429211, 16080822, 26289884, 1367117, 94125317, 7724720, 34395407, 94783382, 49993540, 43008560, 60099262, 34232037, 37954112, 56413080, 54108100, 37882390, 12675084, 72580210, 10634300, 41689463, 22988779, 80687255, 20300706, 32152053, 10103614, 5078185, 81402778, 56911247, 47968955, 42637776, 54600093, 7190580, 14965293, 16288975, 95134084, 14446633, 96139632, 60932111, 60240342, 78585232, 31457126, 48352730, 24296341, 72369756, 56176111, 68307135, 87374656, 96187295, 18035549, 8410200, 66311511, 43667123, 31055633, 22641406, 52849340, 14032825, 19431349, 95892828, 42110693, 97812040, 65511023, 84892748, 51454084, 78207538, 50244666, 73301546, 49921952, 25487422, 78648304, 15210267, 50082551]

# work dir
config['work_dir'] = "./logs/benchmarks/city_scapes_c/pspnet_resnet50/mgda_ub_run_2"
