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
from configs.common.datasets.multi_mnist import config as dataset_config
config.update(dataset_config)

# model config
from configs.common.models.multi_mnist.multi_mnist_lenet5 import model_config_all_tasks as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.mgda_ub import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 19826442
config['train_seeds'] = [39042306, 19772482, 4384863, 79558186, 58249974, 48145345, 81281133, 63005282, 17806831, 84416362, 67267767, 83686992, 49209959, 30467995, 20890238, 60606614, 993892, 9854210, 44076973, 51290708, 61865809, 68931904, 1034798, 67817332, 30832619, 72366083, 45957455, 68016776, 3023074, 82688004, 42752543, 59757953, 8849742, 22648552, 81583412, 78603464, 28453012, 84603712, 11726539, 12703882, 8664415, 1467483, 240935, 7215401, 81370560, 67403550, 3572475, 51660405, 92502197, 27728760, 91151292, 55652707, 33780519, 29088921, 17786271, 35557336, 78437210, 33467978, 8689315, 29784054, 70659981, 48253419, 13131376, 8010427, 48801338, 55649738, 18540213, 37542455, 93294369, 34125, 42952146, 59607496, 68066428, 13183175, 67954700, 60438146, 55079902, 48756981, 58255024, 75006015, 51332213, 3724530, 2392655, 65457106, 80697132, 46196192, 17630694, 57959786, 28320398, 24422631, 5585169, 65920196, 79057260, 66435338, 8443652, 52329871, 28740533, 96550926, 4889620, 88945338]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/mgda_ub_run_0"