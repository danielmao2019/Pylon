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
train_data_cfg['train_dataset']['args']['overlap'] = 0.4
config.update(train_data_cfg)
from configs.common.datasets.point_cloud_registration.val.geotransformer_synth_pcr_data_cfg import data_cfg as val_data_cfg
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
config['init_seed'] = 42867616
config['train_seeds'] = [29272647, 9061603, 73285554, 52940283, 25368006, 91521382, 65349188, 83720260, 99096416, 29501956, 39989776, 18506534, 98347050, 53889170, 94021571, 63862700, 2878739, 42160202, 85036001, 76688546, 92623934, 37945939, 74104625, 33606649, 1355757, 91965342, 13214833, 30657036, 2512483, 55657109, 48585343, 282312, 88084371, 94703260, 28758349, 36128614, 65070721, 98887696, 21029284, 79099170, 78043337, 54363730, 15269466, 26583062, 59485125, 41117259, 37616806, 67732644, 61858134, 55988493, 61285301, 46003827, 95628691, 9116189, 73034008, 36533885, 52835295, 19782805, 23861482, 13952182, 24437380, 12519918, 72754681, 77015964, 71333852, 44286944, 26882454, 10880840, 95509360, 55638463, 88939060, 44332003, 80412401, 73946133, 10435716, 38891334, 44479507, 58349081, 41327845, 40895468, 94785496, 44498347, 2094532, 15661213, 41443710, 2991522, 34810740, 71871480, 3849542, 28657111, 91269110, 76797241, 2916450, 23591688, 62942652, 85209555, 64499634, 8515459, 69743822, 95574465]

# work dir
config['work_dir'] = "./logs/benchmarks/point_cloud_registration/synth_pcr/overlap_0.4/GeoTransformer_run_0"
