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
from configs.common.datasets.multi_task_learning.nyu_v2_f import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_f.pspnet_resnet50 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.multi_task_learning.mgda_ub import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 36729730
config['train_seeds'] = [23406578, 54573804, 62312850, 75028949, 65438465, 53800421, 82899481, 15657811, 87537547, 46349936, 53579507, 39643981, 35186894, 16685823, 41411819, 8221020, 54064037, 90102017, 99146697, 14325706, 98849597, 38607788, 72581730, 40007420, 86150553, 45482178, 89379160, 16265381, 48174436, 99169191, 6295382, 93986588, 79195178, 15520083, 94076717, 46373752, 34729333, 26438120, 19047190, 17532255, 86815053, 7751171, 45612678, 68114207, 84499954, 61306437, 47701487, 38206504, 94485354, 13459361, 46819432, 97734243, 39344265, 35106887, 57663129, 39026366, 4824524, 4849136, 65260270, 91457294, 58249007, 9860476, 27112191, 10853025, 63300295, 31792285, 60379887, 64804532, 65061502, 99804164, 34801390, 10249357, 73302549, 82409595, 2086186, 60863786, 12709960, 26269367, 49417871, 69396656, 30610670, 47875698, 15767159, 4203497, 95311716, 86503210, 3449488, 10632036, 62438338, 91366136, 87604525, 26534359, 50168181, 44267923, 21245630, 28447308, 32884435, 64893771, 1494280, 66892043]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/mgda_ub_run_1"
