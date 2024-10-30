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
from configs.common.optimizers.graddrop import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 18649270
config['train_seeds'] = [43256442, 55700162, 3750920, 18665187, 83344653, 58352208, 26125469, 45893192, 32255854, 2304519, 70754493, 31532227, 68956278, 86942175, 24790220, 8495661, 927041, 30063879, 41149656, 3075369, 36059497, 21859584, 9088006, 29695412, 39261791, 80908265, 73834362, 82470503, 59292165, 30034498, 77192803, 27780650, 43876819, 41794243, 44498212, 5338728, 25136869, 98280676, 71526522, 17737761, 51132017, 3786794, 37977697, 86238825, 48700294, 91108465, 50768977, 32558254, 80761222, 55276470, 89377275, 80807768, 3950687, 51259450, 92975282, 24358118, 50969348, 18869282, 62030864, 62868232, 62067734, 25700212, 54883705, 87842966, 33934136, 11844430, 17428041, 7754344, 71506588, 52570082, 97251480, 18933180, 13525253, 87020606, 87294548, 89103480, 91010094, 16850853, 59780346, 7162711, 85383732, 31781136, 46071723, 85615940, 62627586, 66082495, 8063271, 80444957, 36958821, 78945203, 84558447, 20629839, 52058598, 53278626, 23677743, 85472240, 73112662, 20129952, 20984241, 76510300]

# work dir
config['work_dir'] = "./logs/benchmarks/nyu_v2_f/pspnet_resnet50/graddrop_run_1"