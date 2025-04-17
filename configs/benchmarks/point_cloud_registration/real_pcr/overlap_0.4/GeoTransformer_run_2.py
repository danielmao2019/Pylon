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
from configs.common.datasets.point_cloud_registration.train.geotransformer_real_pcr_data_cfg import data_cfg as train_data_cfg
train_data_cfg['train_dataset']['args']['overlap'] = 0.4
config.update(train_data_cfg)
from configs.common.datasets.point_cloud_registration.val.geotransformer_real_pcr_data_cfg import data_cfg as val_data_cfg
val_data_cfg['val_dataset']['args']['overlap'] = 0.4
config.update(val_data_cfg)

# model config
from configs.common.models.point_cloud_registration.geotransformer_cfg import model_cfg
config['model'] = model_cfg

from configs.common.criteria.point_cloud_registration.geotransformer_criterion_cfg import criterion_cfg
config['criterion'] = criterion_cfg

from configs.common.metrics.point_cloud_registration.geotransformer_metric_cfg import metric_cfg
config['metric'] = metric_cfg

# seeds
config['init_seed'] = 75138615
config['train_seeds'] = [28070660, 95100034, 36551437, 82413597, 65657954, 61093413, 86899016, 90756804, 87454602, 99850464, 11423645, 83531617, 36627920, 7929969, 57404175, 60910663, 89147183, 99629390, 48902177, 30010861, 69579033, 42520227, 36018066, 99531615, 54387700, 34829221, 66595739, 18656925, 91458019, 11589325, 91682069, 55428636, 5900529, 14375033, 48007303, 91655042, 40621953, 65581882, 14756018, 83975015, 50502611, 46529341, 66897989, 7790981, 77054491, 14131425, 60346543, 952812, 16218038, 68915244, 43026693, 80851192, 12797097, 81217922, 70849838, 14293793, 84524642, 60848285, 85518795, 48214663, 25277975, 27280649, 19322776, 96805711, 21085533, 90469629, 42060861, 15772367, 71307139, 54190380, 43530144, 22300258, 48432561, 13833140, 68266507, 3676688, 6387697, 58095587, 220907, 9624979, 81502073, 94494751, 90219961, 63373229, 66746978, 10293630, 29339395, 60907486, 32831657, 77802781, 46845982, 60627634, 66615403, 80832197, 3709951, 75348305, 91401265, 49697952, 28962129, 74222617]

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/real_pcr/overlap_0.4/GeoTransformer_run_2"
