# This file is automatically generated by `./configs/benchmarks/point_cloud_registration/gen.py`.
# Please do not attempt to modify manually.
import torch
import optimizers


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
    'criterion': None,
    'val_dataset': None,
    'val_dataloader': None,
    'test_dataset': None,
    'test_dataloader': None,
    'metric': None,
    # model config
    'model': None,
    # optimizer config
    'optimizer': {
        'class': optimizers.SingleTaskOptimizer,
        'args': {
            'optimizer_config': {
                'class': torch.optim.Adam,
                'args': {
                    'params': None,
                    'lr': 1.0e-4,
                    'weight_decay': 1.0e-06,
                },
            },
        },
    },
    # scheduler config
    'scheduler': {
        'class': torch.optim.lr_scheduler.StepLR,
        'args': {
            'optimizer': None,
            'step_size': 1000,
            'gamma': 0.95,
        },
    },
}

from runners import SupervisedSingleTaskTrainer
config['runner'] = SupervisedSingleTaskTrainer

# data config
from configs.common.datasets.point_cloud_registration.train.geotransformer_synth_pcr_data_cfg import data_cfg as train_data_cfg
train_data_cfg['train_dataset']['args']['overlap'] = 1.0
config.update(train_data_cfg)
from configs.common.datasets.point_cloud_registration.val.geotransformer_synth_pcr_data_cfg import data_cfg as val_data_cfg
val_data_cfg['val_dataset']['args']['overlap'] = 1.0
config.update(val_data_cfg)

# model config
from configs.common.models.point_cloud_registration.geotransformer_cfg import model_cfg
config['model'] = model_cfg

from configs.common.criteria.point_cloud_registration.geotransformer_criterion_cfg import criterion_cfg
config['criterion'] = criterion_cfg

from configs.common.metrics.point_cloud_registration.geotransformer_metric_cfg import metric_cfg
config['metric'] = metric_cfg

# seeds
config['init_seed'] = 1698597
config['train_seeds'] = [34070452, 55981750, 27227594, 38549127, 2958624, 64539565, 96548656, 79006279, 4299880, 22237392, 91261450, 64837544, 5060861, 7211837, 962502, 41772289, 5640335, 86052001, 14572427, 95007891, 89244323, 73700362, 25923293, 75628240, 89117346, 58451713, 44057939, 8909890, 11431784, 75441990, 96512182, 99202249, 42784733, 79767244, 61461065, 79271027, 89430369, 9724344, 24888657, 10183018, 38911033, 57317472, 61325558, 52186304, 70714179, 46795393, 68438144, 68474231, 56571692, 29133570, 69575279, 29886181, 26725888, 82153662, 40400454, 58804033, 49578482, 71174035, 21277637, 34246098, 7752850, 22976584, 97726413, 43197752, 44144001, 16294878, 81963969, 35514610, 31086030, 58421485, 89719919, 41530077, 65324243, 11769047, 11544095, 22861832, 19432562, 76648252, 38014066, 37904132, 54710488, 97157134, 45684888, 41293772, 99869904, 88202190, 96485516, 79345361, 4685821, 71903752, 36162723, 41101989, 84458674, 51819136, 3016321, 81061245, 46807919, 76417361, 80984303, 36639347]

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/synth_pcr/overlap_1.0/GeoTransformer_run_2"
