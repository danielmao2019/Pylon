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
from configs.common.datasets.multi_task_learning.nyu_v2_f import config as dataset_config
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
from configs.common.models.multi_task_learning.nyu_v2_f.pspnet_resnet50 import model_config_normal_estimation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 62254366
config['train_seeds'] = [61595702, 69153724, 71814655, 28480761, 84672019, 81517575, 85853550, 24793278, 79069084, 45450845, 30428424, 39353116, 34101414, 83796337, 92099144, 6798349, 3563991, 32111229, 87973580, 17029170, 3849558, 8231654, 53053096, 24573452, 19208685, 9512772, 24083428, 70184005, 38112508, 62408391, 85524092, 86149330, 19723603, 82666496, 97547327, 82956013, 73674866, 52866020, 25242407, 8716570, 86858063, 18036407, 82898566, 85216393, 50555729, 80104752, 636189, 96289813, 89695514, 49085689, 26963946, 23342195, 39853481, 98811712, 8859462, 70677212, 56258651, 96836028, 29153686, 52718121, 92057139, 45222913, 52195494, 16983339, 2288323, 14020811, 20929313, 97237244, 39821589, 92663178, 59947514, 33331700, 70450742, 28430086, 23170527, 55202642, 13448250, 1354899, 41469988, 57870146, 25655563, 51952589, 88062663, 92009456, 70520079, 98120091, 40821021, 79470588, 95343068, 67349692, 35587429, 42773126, 88499695, 90899756, 40958026, 62908454, 94506667, 28911528, 40203228, 3440665]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/single_task_normal_estimation_run_0"
