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
from configs.common.optimizers.cagrad import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 6920276
config['train_seeds'] = [65127971, 53008299, 63809159, 89095630, 42718744, 16366219, 4358185, 36584269, 30702553, 20882960, 96189751, 61047439, 41110638, 48944144, 77079451, 1010739, 4947254, 15134153, 43391692, 91597389, 74791028, 68862912, 94218122, 5128234, 36174271, 16270730, 90177694, 24764065, 30752815, 99453646, 49033877, 24341069, 43313670, 46730722, 34332002, 40296447, 35332151, 59611664, 32098860, 57416844, 24868456, 59811572, 24765263, 25726362, 6966887, 45829871, 60168013, 25846295, 3201677, 5060299, 8238918, 42243638, 71822406, 73938282, 35178304, 75390635, 99765076, 96259909, 14171442, 18202738, 20155714, 95639506, 19812420, 5474329, 92159939, 18918661, 85045879, 34336396, 64407528, 8797839, 57906297, 55466915, 65490402, 76499022, 33635003, 27330789, 53096561, 61047892, 67926804, 79165965, 35219674, 58223331, 68961377, 2887178, 67762588, 95135441, 16428927, 49175502, 13534167, 20560096, 86754067, 4499188, 90624507, 99885815, 37764542, 28678025, 45830877, 35449506, 48907648, 8819673]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/cagrad_run_0"
