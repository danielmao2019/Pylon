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
from configs.common.optimizers.pcgrad import optimizer_config
from configs.common.optimizers._core_ import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 81098307
config['train_seeds'] = [33619803, 8260174, 10375434, 67888695, 23637253, 40109929, 38477128, 24625901, 22034777, 74381035, 52462340, 94875275, 52440882, 25998743, 60538782, 39224961, 61397991, 2251435, 94181326, 11836871, 40110641, 67712679, 42346503, 90634780, 48543480, 5582864, 11017959, 46023365, 14192239, 27109384, 44825077, 42397266, 51199487, 33296760, 99109286, 69340927, 4882590, 77595052, 96872544, 24723349, 9206564, 16462958, 3333011, 32208407, 79492527, 98339691, 6362730, 21367004, 49979212, 69017202, 82665953, 34105099, 94258648, 36974913, 93392919, 46950026, 35274553, 89555487, 59134589, 93902325, 69820898, 17599731, 33686070, 13534575, 89852570, 99766804, 73134381, 90893037, 33029226, 92114518, 80182225, 48296938, 75255935, 55259674, 20743167, 4097710, 62815456, 84447642, 8256476, 56313342, 16680825, 86885494, 71795928, 64741700, 8317418, 277001, 96040863, 78924347, 68015217, 2284280, 50106420, 29646136, 89437864, 22503825, 71888429, 85911143, 18194782, 54165698, 90118324, 78182443]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/pcgrad_run_1"
