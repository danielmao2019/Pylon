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
from configs.common.datasets.multi_task_learning.city_scapes_c import config as dataset_config
for key in ['train_dataset', 'val_dataset', 'test_dataset']:
    dataset_config[key] = {
        'class': data.datasets.ProjectionDatasetWrapper,
        'args': {
            'dataset_config': dataset_config[key],
            'mapping': {
                'inputs': ['image'],
                'labels': ['semantic_segmentation'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['semantic_segmentation']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['semantic_segmentation']
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.city_scapes_c.pspnet_resnet50 import model_config_semantic_segmentation as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 35340449
config['train_seeds'] = [84580868, 15817166, 94034109, 51358146, 8880457, 91132163, 81423541, 20841879, 48517678, 92311245, 36741241, 97261842, 32317382, 26468807, 61632407, 13604628, 38504733, 38198145, 33986277, 32032472, 88391097, 30432013, 3474217, 47514921, 70242375, 60235470, 38227516, 16280431, 68930960, 93937764, 46670523, 77289627, 45454203, 79316356, 57074511, 19083510, 33878192, 64259111, 73994859, 26677384, 61103448, 28239600, 49373672, 58744352, 41814481, 32696707, 27890492, 25322637, 22811474, 11131738, 1073516, 24168737, 94509260, 5821548, 86813944, 89622999, 8896136, 81277340, 30415340, 61671710, 57362214, 80807613, 78791586, 75987756, 13738115, 22514837, 22557738, 10495234, 40397696, 70398490, 20953355, 44264873, 86750159, 60293209, 19288322, 45253981, 71992053, 39148074, 33234753, 49167011, 20043628, 23645811, 35252949, 48295391, 97723579, 15835036, 43944761, 61429558, 48284745, 87096966, 17140624, 75509573, 24360418, 8569886, 36957123, 22396154, 17745249, 29597285, 78418076, 98489211]

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/city_scapes_c/pspnet_resnet50/single_task_semantic_segmentation_run_2"
