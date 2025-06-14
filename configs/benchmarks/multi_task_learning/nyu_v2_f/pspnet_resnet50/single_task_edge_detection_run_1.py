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
    'val_seeds': None,
    'test_seed': None,
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
                'class': schedulers.lr_lambdas.WarmupLambda,
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
                'labels': ['edge_detection'],
                'meta_info': ['image_filepath', 'image_resolution'],
            },
        },
    }
dataset_config['criterion'] = dataset_config['criterion']['args']['criterion_configs']['edge_detection']
dataset_config['metric'] = dataset_config['metric']['args']['metric_configs']['edge_detection']
config.update(dataset_config)

# model config
from configs.common.models.multi_task_learning.nyu_v2_f.pspnet_resnet50 import model_config_edge_detection as model_config
config['model'] = model_config

# optimizer config
from configs.common.optimizers.single_task_optimizer import single_task_optimizer_config as optimizer_config
from configs.common.optimizers.standard import adam_optimizer_config
optimizer_config['args']['optimizer_config'] = adam_optimizer_config
config['optimizer'] = optimizer_config

# seeds
config['init_seed'] = 61700937
config['train_seeds'] = [11003770, 45411644, 55447403, 90585284, 62091673, 95625907, 12278634, 20535667, 73537582, 63183382, 99498400, 84792205, 14413848, 55547094, 99974009, 86582824, 35954897, 72182678, 17855836, 76914237, 79858099, 45007157, 50361784, 58787765, 31549357, 45743101, 81283811, 49999054, 52249004, 14924982, 13304574, 29976679, 57966396, 66114319, 24295786, 63975323, 79602859, 6888166, 27115190, 41504573, 78923920, 63622122, 41360904, 6157524, 75543026, 2940087, 16414926, 23258336, 24872270, 65507706, 40364780, 86268222, 22546273, 94328875, 33320456, 44156438, 55596179, 32003431, 90388810, 26378807, 84284822, 66334718, 74870961, 1023474, 1033064, 34444345, 41538544, 19800697, 30967114, 13379310, 50220510, 44077159, 88985913, 21464278, 51835469, 1569165, 31027733, 37092442, 7171130, 58316012, 22500167, 14732986, 86827386, 59247886, 5677137, 75142123, 35267255, 17874781, 14815397, 20206131, 22055559, 43242308, 58465684, 49560299, 97479475, 71869220, 53034375, 83422402, 3036838, 15869976]
config['val_seeds'] = [53508827, 2220304, 87352218, 23063903, 10515743, 94698266, 4834201, 95830767, 65581535, 2833131, 89619866, 47257156, 25358257, 56896736, 12336198, 88831370, 82342415, 56057682, 33409299, 47778513, 92448787, 9923949, 46342193, 62956915, 68774915, 24720596, 76444201, 17534757, 27892640, 96639781, 13586644, 85874393, 68591004, 64249799, 70794111, 23738141, 4927122, 77011257, 35337896, 54406330, 5609737, 20553516, 18993395, 24926737, 98082948, 30117589, 45371003, 83345208, 49745060, 51809949, 43785818, 13629835, 17650379, 91762854, 81390475, 38365792, 17615211, 28417288, 50354260, 80942635, 5096031, 5584628, 477951, 65436592, 79220516, 60173834, 58826838, 84051853, 51965332, 79369991, 95783661, 79490129, 28996999, 78078679, 91129294, 85053256, 87428917, 20434343, 15430993, 16749489, 25772000, 29837372, 50661858, 63229490, 47951951, 27689527, 43270030, 77036880, 96812142, 80231279, 97096446, 39714852, 79863406, 60315938, 7578575, 84741925, 9960238, 66886036, 71294073, 99037469]
config['test_seed'] = 13499473

# work dir
config['work_dir'] = "./logs/benchmarks/multi_task_learning/nyu_v2_f/pspnet_resnet50/single_task_edge_detection_run_1"
