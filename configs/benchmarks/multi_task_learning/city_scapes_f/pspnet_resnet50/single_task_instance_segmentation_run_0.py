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
from configs.common.datasets.multi_task_learning.city_scapes_f import config as dataset_config
for key in ['train_dataset', 'val_dataset', 'test_dataset']:
    dataset_config[key] = {
        'class': data.datasets.ProjectionDatasetWrapper,
        'args': {
            'dataset_config': dataset_config[key],
            'mapping': {
                'inputs': ['image'],
                'labels': ['instance_segmentation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['instance_segmentation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['instance_segmentation']
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_f.pspnet_resnet50 import model_config_instance_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 58543233
config['train_seeds'] = [9376171, 58818994, 75468140, 37330437, 6712509, 28752564, 21976840, 69718079, 44617071, 12638622, 40688215, 17986226, 76110137, 21952717, 48778262, 12245372, 17500515, 87149158, 66062215, 88099917, 74818527, 81383898, 58277047, 16839016, 67107021, 32834559, 89827374, 39897706, 50919888, 40485825, 93217326, 20429091, 47892774, 17563976, 74917971, 59583701, 72013112, 87149410, 85868410, 39008160, 15245643, 99972592, 68964783, 65195414, 29453589, 43724879, 99310173, 75522628, 36612306, 18277595, 17939019, 67406722, 83443609, 45325447, 97540926, 24375880, 33021889, 12917244, 95531434, 95960185, 14096728, 25128957, 37349852, 50843529, 94707179, 76855488, 2895349, 89068476, 11957220, 60260396, 97667912, 13372009, 45191168, 74741690, 43967042, 4628771, 3424944, 14879227, 13224892, 59153154, 19648046, 9346205, 62122637, 67606815, 97976408, 43557069, 67614935, 87626931, 39678533, 88379727, 7327218, 67242137, 41839085, 36354639, 87500629, 43051570, 29930232, 47855807, 13499039, 14259895]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_f/pspnet_resnet50/single_task_instance_segmentation_run_0"
