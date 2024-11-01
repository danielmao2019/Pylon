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
from configs.common.datasets.celeb_a import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.celeb_a.celeb_a_resnet18 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.imtl import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 85730158
config['train_seeds'] = [5084629, 21010505, 86718791, 15260330, 75631594, 16153030, 48705482, 21249716, 37417723, 92460432, 90810135, 353453, 62228801, 52301727, 98704280, 19917296, 44428214, 51758446, 92351039, 73217984, 67207922, 94167714, 85707274, 63952469, 13031517, 56565363, 82109817, 2058409, 93037650, 46780437, 36699010, 82442585, 71552436, 9229184, 14675737, 78117766, 47940294, 51682004, 56945064, 66294320, 17947363, 36884761, 29048677, 43492419, 32960430, 51245841, 21370336, 99928811, 93831709, 15160040, 67590718, 7907859, 41390558, 12205366, 49133066, 43731871, 45321519, 72472482, 43591594, 63840327, 6179962, 48859421, 92854778, 52165273, 64772241, 54987688, 69553119, 98276056, 56639019, 51101579, 31815658, 99851901, 11693208, 75198190, 47318453, 23380546, 52456331, 32358286, 75954934, 82162518, 45683084, 10826571, 33135369, 32439943, 80089362, 34401033, 44134565, 98339516, 22616668, 43454590, 37774344, 87274810, 23941420, 66109181, 33574880, 44397846, 43298644, 94461856, 8758503, 2233966]

# work dir
config['work_dir'] = "./logs/benchmarks/celeb_a/celeb_a_resnet18/imtl_run_2"