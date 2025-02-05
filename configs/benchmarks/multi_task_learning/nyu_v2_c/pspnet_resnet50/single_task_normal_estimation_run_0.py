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

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# dataset config
import data
from configs.common.datasets.multi_task_learning.nyu_v2_c import config as dataset_config
for key in ['train_dataset', 'val_dataset', 'test_dataset']:
    dataset_config[key] = {
        'class': data.datasets.ProjectionDatasetWrapper,
        'args': {
            'dataset_config': dataset_config[key],
            'mapping': {
                'inputs': ['image'],
                'labels': ['normal_estimation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['normal_estimation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['normal_estimation']
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_c.pspnet_resnet50 import model_config_normal_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 58673098
config['train_seeds'] = [80079346, 1369738, 82873455, 24283851, 99999770, 38571135, 12760007, 27944487, 80910749, 27609137, 63331612, 24692144, 25957638, 18459486, 2805436, 88624356, 91762913, 79670216, 43933211, 85662943, 17275830, 20133054, 57442398, 77918839, 52510941, 40008479, 62950700, 62015882, 78971783, 23291629, 36625953, 44467328, 96263123, 90932884, 61440932, 6549507, 55243610, 53988752, 84779893, 26266192, 82101527, 51515330, 74168026, 21614090, 41016931, 11147415, 74547400, 92869770, 38098885, 58710208, 71068406, 39625306, 88409357, 7720150, 81366809, 2280819, 91379453, 34606593, 6632631, 56769916, 75159131, 24112450, 26401292, 47326571, 92202271, 27413851, 32882676, 79113787, 116269, 90354251, 68895494, 61678317, 5356459, 3237179, 99272026, 12617842, 73199639, 7184703, 14960980, 42689757, 59405302, 55598298, 81857792, 25198328, 59245864, 62213132, 24527092, 99081964, 12487478, 12255678, 12610631, 64242231, 39386867, 54547008, 66054997, 48113287, 55376614, 616192, 65195701, 30824691]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_c/pspnet_resnet50/single_task_normal_estimation_run_0"
