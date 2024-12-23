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
config['init_seed'] = 59528299
config['train_seeds'] = [12858802, 58578000, 75800226, 1235961, 70285856, 2337130, 20108683, 16276385, 11800534, 28923326, 78283427, 68591954, 17492401, 86589816, 92260760, 53874726, 30854659, 2992559, 52173298, 92683634, 91951228, 40021696, 69801571, 15902397, 79511163, 99375155, 4656137, 77534814, 40438630, 37534397, 98856536, 7095195, 71116445, 57515492, 66650018, 96506612, 25273241, 76635011, 63738372, 8663793, 1833309, 5646063, 48471483, 28818992, 51319933, 85238517, 45404634, 57680218, 42864709, 83340811, 89220738, 31826702, 62632746, 433338, 78968472, 27889410, 24430278, 19744330, 61324428, 54647121, 7428046, 17330735, 47967401, 96767824, 61389926, 63134278, 31092131, 89579715, 84171086, 38130289, 42513924, 44269159, 38041470, 50896105, 16872442, 61761323, 97236879, 55366552, 98620163, 23243137, 59048540, 71433785, 3462058, 93155737, 8415736, 70826470, 10893826, 77921078, 78260150, 90843408, 14859580, 98860581, 77545621, 20980990, 71383916, 79457692, 39947795, 14049626, 11654257, 2771673]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_mnist/multi_mnist_lenet5/pcgrad_run_0"
