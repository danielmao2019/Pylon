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
from configs.common.optimizers.mgda import optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
optimizer_config['args']['per_layer'] = False
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 80156307
config['train_seeds'] = [19619889, 7998907, 98791131, 84889459, 81073940, 55520540, 93222682, 42406499, 65711637, 70665048, 91749938, 66485171, 42674371, 17196978, 64072713, 76543545, 45538775, 42225908, 20278092, 99383377, 82594181, 42753933, 67834874, 97683334, 66237764, 96449166, 16768303, 82144930, 9183794, 60468024, 20770844, 64187980, 75378196, 39592378, 8192061, 28107573, 25955135, 28816286, 88472170, 38221095, 30888239, 36546697, 28910563, 66653434, 51054461, 37399328, 73091621, 95104219, 27617163, 49586444, 43159277, 85183076, 56046216, 56845350, 27538948, 78035628, 95886352, 48050806, 84548717, 26352643, 50884983, 98275556, 39803463, 99884683, 1256922, 69519847, 23303727, 44880074, 57356261, 52513834, 32028579, 85341264, 64069540, 32676581, 81204725, 19417938, 95137751, 99982358, 4997391, 90491533, 50733348, 50896688, 17151841, 62452591, 96199066, 32094488, 41278792, 20607163, 99554845, 37154673, 11813627, 59102978, 86780571, 51196312, 18114790, 58991263, 56227765, 6186061, 38916536, 15144427]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/mgda_run_1"
