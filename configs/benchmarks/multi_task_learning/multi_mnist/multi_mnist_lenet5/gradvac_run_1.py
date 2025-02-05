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
from configs.common.optimizers.gradvac import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 65134171
config['train_seeds'] = [45655697, 38408721, 85412260, 62811501, 26651900, 18129094, 37993498, 55919171, 78265306, 81405253, 83758299, 76144737, 9425438, 28400800, 61016198, 94026386, 34807064, 63114023, 31097234, 86054661, 38964777, 79553065, 31512369, 65913928, 13334783, 20052430, 91572353, 75160242, 53239122, 43820889, 50985195, 33625874, 88404316, 54451975, 64462292, 62313811, 15909809, 29200582, 69347296, 34685686, 95338230, 87480930, 36863141, 18014620, 60422418, 70058143, 16402603, 77850909, 4820069, 87245091, 6264029, 27231571, 66892767, 57899368, 51965789, 26175515, 64444174, 64237770, 89445035, 24684833, 77962842, 15985952, 63262466, 44818861, 58866253, 52219623, 83182431, 66006534, 31375008, 76774361, 48951961, 32643786, 6169538, 15690544, 63695148, 49239836, 77235904, 38977137, 35211188, 7888746, 92027495, 32010604, 17480231, 73536386, 2014653, 89634341, 44700886, 66767517, 10384546, 9968859, 22023845, 5696167, 18361790, 68018873, 52585570, 32882061, 64390047, 68682645, 80271492, 21438344]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/gradvac_run_1"
